package com.yourcompany.sensorspoke.sensors.thermal.tc001

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.max
import kotlin.math.min

/**
 * TC001PerformanceMonitor - Advanced performance monitoring for thermal system
 *
 * Provides comprehensive performance analysis for TC001 thermal integration:
 * - Real-time frame rate monitoring with statistics
 * - Temperature processing latency analysis
 * - Memory usage tracking for thermal data
 * - Connection quality assessment
 * - System health diagnostics
 * - Performance alerts and recommendations
 */
class TC001PerformanceMonitor(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TC001PerformanceMonitor"
        private const val PERFORMANCE_WINDOW_SIZE = 100 // Analyze last 100 frames
        private const val ALERT_THRESHOLD_LOW_FPS = 20.0 // Below 20 FPS
        private const val ALERT_THRESHOLD_HIGH_LATENCY = 100.0 // Above 100ms processing time
        private const val MEMORY_WARNING_THRESHOLD_MB = 50.0 // 50MB thermal data in memory
    }

    // Performance metrics storage
    private val frameTimestamps = ConcurrentLinkedQueue<Long>()
    private val processingLatencies = ConcurrentLinkedQueue<Long>()
    private val memoryUsages = ConcurrentLinkedQueue<Long>()
    private val temperatureReadings = ConcurrentLinkedQueue<Float>()

    // Performance state
    private val _performanceMetrics = MutableLiveData<TC001PerformanceMetrics>()
    val performanceMetrics: LiveData<TC001PerformanceMetrics> = _performanceMetrics

    private val _performanceAlerts = MutableLiveData<List<TC001PerformanceAlert>>()
    val performanceAlerts: LiveData<List<TC001PerformanceAlert>> = _performanceAlerts

    private val _systemHealth = MutableLiveData<TC001SystemHealth>()
    val systemHealth: LiveData<TC001SystemHealth> = _systemHealth

    private var monitoringJob: Job? = null
    private var isMonitoring = false
    private var sessionStartTime = 0L

    /**
     * Start performance monitoring for TC001 system
     */
    fun startMonitoring() {
        if (isMonitoring) return

        isMonitoring = true
        sessionStartTime = SystemClock.elapsedRealtimeNanos()

        monitoringJob =
            CoroutineScope(Dispatchers.IO).launch {
                while (isMonitoring) {
                    try {
                        updatePerformanceMetrics()
                        checkPerformanceAlerts()
                        assessSystemHealth()
                        delay(1000) // Update every second
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in performance monitoring", e)
                    }
                }
            }

        Log.i(TAG, "TC001 performance monitoring started")
    }

    /**
     * Stop performance monitoring
     */
    fun stopMonitoring() {
        isMonitoring = false
        monitoringJob?.cancel()

        // Generate final performance report
        generatePerformanceReport()

        Log.i(TAG, "TC001 performance monitoring stopped")
    }

    /**
     * Record thermal frame processing event
     */
    fun recordFrameProcessed() {
        val timestamp = SystemClock.elapsedRealtimeNanos()
        frameTimestamps.offer(timestamp)

        // Keep only recent frames for analysis
        while (frameTimestamps.size > PERFORMANCE_WINDOW_SIZE) {
            frameTimestamps.poll()
        }
    }

    /**
     * Record thermal data processing latency
     */
    fun recordProcessingLatency(
        startTime: Long,
        endTime: Long,
    ) {
        val latency = (endTime - startTime) / 1_000_000 // Convert to milliseconds
        processingLatencies.offer(latency)

        while (processingLatencies.size > PERFORMANCE_WINDOW_SIZE) {
            processingLatencies.poll()
        }
    }

    /**
     * Record memory usage for thermal data
     */
    fun recordMemoryUsage(bytesUsed: Long) {
        memoryUsages.offer(bytesUsed)

        while (memoryUsages.size > PERFORMANCE_WINDOW_SIZE) {
            memoryUsages.poll()
        }
    }

    /**
     * Record temperature reading for statistical analysis
     */
    fun recordTemperatureReading(temperature: Float) {
        temperatureReadings.offer(temperature)

        while (temperatureReadings.size > PERFORMANCE_WINDOW_SIZE) {
            temperatureReadings.poll()
        }
    }

    /**
     * Calculate and update comprehensive performance metrics
     */
    private fun updatePerformanceMetrics() {
        val currentTime = SystemClock.elapsedRealtimeNanos()

        // Calculate frame rate
        val frameRate = calculateFrameRate()

        // Calculate average processing latency
        val avgLatency =
            processingLatencies.toList().let { latencies ->
                if (latencies.isNotEmpty()) latencies.average() else 0.0
            }

        // Calculate memory usage statistics
        val memoryStats = calculateMemoryStats()

        // Calculate temperature statistics
        val tempStats = calculateTemperatureStats()

        // Calculate session uptime
        val sessionUptime = (currentTime - sessionStartTime) / 1_000_000_000.0 // seconds

        val metrics =
            TC001PerformanceMetrics(
                frameRate = frameRate,
                averageProcessingLatency = avgLatency,
                memoryUsageMB = memoryStats.currentMB,
                peakMemoryUsageMB = memoryStats.peakMB,
                sessionUptime = sessionUptime,
                totalFramesProcessed = frameTimestamps.size,
                temperatureStability = tempStats.stability,
                avgTemperature = tempStats.average,
                connectionQuality = assessConnectionQuality(),
                systemLoad = calculateSystemLoad(),
            )

        _performanceMetrics.postValue(metrics)
    }

    /**
     * Calculate current frame rate
     */
    private fun calculateFrameRate(): Double {
        val timestamps = frameTimestamps.toList()
        if (timestamps.size < 2) return 0.0

        val timeSpan = (timestamps.last() - timestamps.first()) / 1_000_000_000.0 // seconds
        return if (timeSpan > 0) (timestamps.size - 1) / timeSpan else 0.0
    }

    /**
     * Calculate memory usage statistics
     */
    private fun calculateMemoryStats(): MemoryStats {
        val usages = memoryUsages.toList()
        if (usages.isEmpty()) return MemoryStats(0.0, 0.0)

        val currentMB = usages.lastOrNull()?.let { it / (1024.0 * 1024.0) } ?: 0.0
        val peakMB = usages.maxOrNull()?.let { it / (1024.0 * 1024.0) } ?: 0.0

        return MemoryStats(currentMB, peakMB)
    }

    /**
     * Calculate temperature reading statistics
     */
    private fun calculateTemperatureStats(): TemperatureStats {
        val readings = temperatureReadings.toList()
        if (readings.isEmpty()) return TemperatureStats(0.0, 0.0)

        val average = readings.average()
        val variance = readings.map { (it - average) * (it - average) }.average()
        val stability = max(0.0, 100.0 - variance) // Higher is more stable

        return TemperatureStats(average, stability)
    }

    /**
     * Assess TC001 connection quality
     */
    private fun assessConnectionQuality(): Double {
        val frameRate = calculateFrameRate()
        val avgLatency =
            processingLatencies.toList().let {
                if (it.isNotEmpty()) it.average() else 0.0
            }

        // Connection quality based on frame rate and latency
        val frameRateScore = min(100.0, (frameRate / 25.0) * 100.0) // Target 25 FPS
        val latencyScore = max(0.0, 100.0 - (avgLatency / 2.0)) // Lower latency is better

        return (frameRateScore + latencyScore) / 2.0
    }

    /**
     * Calculate current system load
     */
    private fun calculateSystemLoad(): Double {
        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val usedMemory = totalMemory - freeMemory

        return (usedMemory.toDouble() / maxMemory.toDouble()) * 100.0
    }

    /**
     * Check for performance issues and generate alerts
     */
    private fun checkPerformanceAlerts() {
        val alerts = mutableListOf<TC001PerformanceAlert>()

        // Check frame rate
        val frameRate = calculateFrameRate()
        if (frameRate < ALERT_THRESHOLD_LOW_FPS) {
            alerts.add(
                TC001PerformanceAlert(
                    type = TC001AlertType.LOW_FRAME_RATE,
                    severity = TC001AlertSeverity.WARNING,
                    message = "Frame rate below threshold: ${String.format("%.1f", frameRate)} FPS",
                    recommendation = "Check USB connection and system resources",
                ),
            )
        }

        // Check processing latency
        val avgLatency =
            processingLatencies.toList().let {
                if (it.isNotEmpty()) it.average() else 0.0
            }
        if (avgLatency > ALERT_THRESHOLD_HIGH_LATENCY) {
            alerts.add(
                TC001PerformanceAlert(
                    type = TC001AlertType.HIGH_LATENCY,
                    severity = TC001AlertSeverity.WARNING,
                    message = "Processing latency high: ${String.format("%.1f", avgLatency)} ms",
                    recommendation = "Close background apps and ensure sufficient system resources",
                ),
            )
        }

        // Check memory usage
        val memoryStats = calculateMemoryStats()
        if (memoryStats.currentMB > MEMORY_WARNING_THRESHOLD_MB) {
            alerts.add(
                TC001PerformanceAlert(
                    type = TC001AlertType.HIGH_MEMORY_USAGE,
                    severity = TC001AlertSeverity.INFO,
                    message = "High memory usage: ${String.format("%.1f", memoryStats.currentMB)} MB",
                    recommendation = "Normal for thermal processing, monitor for leaks",
                ),
            )
        }

        _performanceAlerts.postValue(alerts)
    }

    /**
     * Assess overall TC001 system health
     */
    private fun assessSystemHealth() {
        val frameRate = calculateFrameRate()
        val connectionQuality = assessConnectionQuality()
        val systemLoad = calculateSystemLoad()

        val health =
            when {
                frameRate > 20 && connectionQuality > 80 && systemLoad < 70 -> TC001HealthStatus.EXCELLENT
                frameRate > 15 && connectionQuality > 60 && systemLoad < 85 -> TC001HealthStatus.GOOD
                frameRate > 10 && connectionQuality > 40 && systemLoad < 95 -> TC001HealthStatus.FAIR
                else -> TC001HealthStatus.POOR
            }

        val systemHealth =
            TC001SystemHealth(
                status = health,
                frameRate = frameRate,
                connectionQuality = connectionQuality,
                systemLoad = systemLoad,
                recommendation = getHealthRecommendation(health),
            )

        _systemHealth.postValue(systemHealth)
    }

    /**
     * Get health recommendation based on status
     */
    private fun getHealthRecommendation(status: TC001HealthStatus): String =
        when (status) {
            TC001HealthStatus.EXCELLENT -> "System performing optimally"
            TC001HealthStatus.GOOD -> "System performing well"
            TC001HealthStatus.FAIR -> "Consider closing background apps"
            TC001HealthStatus.POOR -> "Check USB connection and system resources"
        }

    /**
     * Generate comprehensive performance report
     */
    private fun generatePerformanceReport() {
        val sessionDuration = (SystemClock.elapsedRealtimeNanos() - sessionStartTime) / 1_000_000_000.0
        val finalFrameRate = calculateFrameRate()
        val finalConnectionQuality = assessConnectionQuality()

        Log.i(
            TAG,
            """
            |TC001 Performance Report:
            |Session Duration: ${String.format("%.1f", sessionDuration)}s
            |Total Frames: ${frameTimestamps.size}
            |Average Frame Rate: ${String.format("%.1f", finalFrameRate)} FPS
            |Connection Quality: ${String.format("%.1f", finalConnectionQuality)}%
            |Peak Memory Usage: ${String.format("%.1f", calculateMemoryStats().peakMB)} MB
            """.trimMargin(),
        )
    }
}

// Supporting data classes
data class TC001PerformanceMetrics(
    val frameRate: Double,
    val averageProcessingLatency: Double,
    val memoryUsageMB: Double,
    val peakMemoryUsageMB: Double,
    val sessionUptime: Double,
    val totalFramesProcessed: Int,
    val temperatureStability: Double,
    val avgTemperature: Double,
    val connectionQuality: Double,
    val systemLoad: Double,
)

data class TC001PerformanceAlert(
    val type: TC001AlertType,
    val severity: TC001AlertSeverity,
    val message: String,
    val recommendation: String,
)

data class TC001SystemHealth(
    val status: TC001HealthStatus,
    val frameRate: Double,
    val connectionQuality: Double,
    val systemLoad: Double,
    val recommendation: String,
)

private data class MemoryStats(
    val currentMB: Double,
    val peakMB: Double,
)

private data class TemperatureStats(
    val average: Double,
    val stability: Double,
)

enum class TC001AlertType {
    LOW_FRAME_RATE,
    HIGH_LATENCY,
    HIGH_MEMORY_USAGE,
    CONNECTION_UNSTABLE,
    TEMPERATURE_ANOMALY,
}

enum class TC001AlertSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
}

enum class TC001HealthStatus {
    EXCELLENT,
    GOOD,
    FAIR,
    POOR,
}
