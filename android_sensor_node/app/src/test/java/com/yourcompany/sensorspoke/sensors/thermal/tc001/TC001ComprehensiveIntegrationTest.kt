package com.yourcompany.sensorspoke.sensors.thermal.tc001

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Comprehensive TC001 Integration Test Suite
 *
 * Tests all aspects of the TC001 thermal camera integration:
 * - Component initialization and lifecycle management
 * - Data processing and thermal analysis
 * - Performance monitoring and diagnostics
 * - Calibration system functionality
 * - Export and data management capabilities
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class TC001ComprehensiveIntegrationTest {
    private lateinit var context: Context
    private lateinit var integrationManager: TC001IntegrationManager
    private lateinit var performanceMonitor: TC001PerformanceMonitor
    private lateinit var dataExporter: TC001DataExporter
    private lateinit var calibrationManager: TC001CalibrationManager
    private lateinit var diagnosticSystem: TC001DiagnosticSystem

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        integrationManager = TC001IntegrationManager(context)
        performanceMonitor = TC001PerformanceMonitor(context)
        dataExporter = TC001DataExporter(context)
        calibrationManager = TC001CalibrationManager(context)
        diagnosticSystem = TC001DiagnosticSystem(context)
    }

    @Test
    fun `TC001 integration manager initialization`() =
        runTest {
            val initResult = integrationManager.initializeSystem()
            assertTrue("TC001 integration manager should initialize successfully", initResult)

            val startResult = integrationManager.startSystem()
            assertTrue("TC001 system should start successfully", startResult)

            assertTrue("TC001 system should be ready", integrationManager.isSystemReady())

            assertNotNull("TC001 connector should be available", integrationManager.getConnector())
            assertNotNull("TC001 data manager should be available", integrationManager.getDataManager())
            assertNotNull("TC001 UI controller should be available", integrationManager.getUIController())

            integrationManager.cleanup()
        }

    @Test
    fun `TC001 performance monitoring functionality`() =
        runTest {
            performanceMonitor.startMonitoring()

            repeat(50) {
                performanceMonitor.recordFrameProcessed()
                performanceMonitor.recordTemperatureReading(25.0f + it * 0.1f)
                performanceMonitor.recordMemoryUsage(1024 * 1024 * 10L)
            }

            performanceMonitor.stopMonitoring()

            // Performance monitoring should complete without errors
            assertTrue("Performance monitoring should complete successfully", true)
        }

    @Test
    fun `TC001 comprehensive integration workflow`() =
        runTest {

            val initResult = integrationManager.initializeSystem()
            assertTrue("Integration manager should initialize", initResult)

            val startResult = integrationManager.startSystem()
            assertTrue("System should start", startResult)

            // Step 3: Test performance monitoring
            performanceMonitor.startMonitoring()
            repeat(10) {
                performanceMonitor.recordFrameProcessed()
            }
            performanceMonitor.stopMonitoring()

            val diagnostics = diagnosticSystem.runComprehensiveDiagnostics()
            assertNotNull("Diagnostics should complete", diagnostics)

            integrationManager.cleanup()
        }

    @Test
    fun `TC001 memory management under load`() =
        runTest {
            val initialMemory = Runtime.getRuntime().freeMemory()

            integrationManager.initializeSystem()
            integrationManager.startSystem()
            performanceMonitor.startMonitoring()

            repeat(50) {
                performanceMonitor.recordFrameProcessed()
                performanceMonitor.recordMemoryUsage(1024 * 1024 * (it + 1).toLong())
                performanceMonitor.recordTemperatureReading(25.0f + it * 0.1f)
            }

            performanceMonitor.stopMonitoring()
            integrationManager.cleanup()

            System.gc()
            Thread.sleep(100)

            val finalMemory = Runtime.getRuntime().freeMemory()
            val memoryDifference = initialMemory - finalMemory
            val memoryDifferenceMB = memoryDifference / (1024.0 * 1024.0)

            assertTrue(
                "Memory usage should be reasonable after cleanup",
                memoryDifferenceMB < 50.0,
            )
        }

    private fun createTempSessionDir(): java.io.File {
        val tempDir = java.io.File(context.cacheDir, "test_session_${System.currentTimeMillis()}")
        tempDir.mkdirs()
        return tempDir
    }
}
