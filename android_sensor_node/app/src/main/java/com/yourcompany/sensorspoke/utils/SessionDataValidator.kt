package com.yourcompany.sensorspoke.utils

import android.os.StatFs
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.*

/**
 * Utility class for session data validation and management
 * Provides functionality for validating data integrity, timestamp consistency, and storage management
 */
object SessionDataValidator {
    private const val TAG = "SessionDataValidator"

    /**
     * Validate a completed session directory for data integrity
     */
    fun validateSessionData(sessionDir: File): ValidationResult {
        val results = mutableMapOf<String, Boolean>()
        val issues = mutableListOf<String>()

        try {
            val metadataFile = File(sessionDir, "session_metadata.json")
            if (metadataFile.exists()) {
                results["metadata"] = true
                validateMetadata(metadataFile, issues)
            } else {
                results["metadata"] = false
                issues.add("Missing session metadata file")
            }

            val subdirs = sessionDir.listFiles { file -> file.isDirectory }
            subdirs?.forEach { subdir ->
                when (subdir.name.lowercase()) {
                    "rgb" -> validateRgbData(subdir, results, issues)
                    "thermal" -> validateThermalData(subdir, results, issues)
                    "gsr" -> validateGsrData(subdir, results, issues)
                    "audio" -> validateAudioData(subdir, results, issues)
                }
            }

            val csvFiles = sessionDir.walkTopDown()
                .filter { it.isFile && it.name.endsWith(".csv") }
                .toList()

            if (csvFiles.isNotEmpty()) {
                validateTimestampConsistency(csvFiles, results, issues)
            }

            Log.i(TAG, "Session validation completed: ${sessionDir.name} - ${issues.size} issues found")
        } catch (e: Exception) {
            Log.e(TAG, "Error validating session data: ${e.message}", e)
            issues.add("Validation error: ${e.message}")
        }

        return ValidationResult(
            sessionId = sessionDir.name,
            isValid = issues.isEmpty(),
            results = results,
            issues = issues,
        )
    }

    private fun validateMetadata(metadataFile: File, issues: MutableList<String>) {
        try {
            val content = metadataFile.readText()
            val metadata = JSONObject(content)

            val requiredFields = listOf(
                "session_id",
                "start_timestamp_ms",
                "start_timestamp_ns",
                "recorders",
                "session_status",
            )

            requiredFields.forEach { field ->
                if (!metadata.has(field)) {
                    issues.add("Missing required metadata field: $field")
                }
            }

            if (metadata.has("start_timestamp_ms") && metadata.has("start_timestamp_ns")) {
                val timestampMs = metadata.getLong("start_timestamp_ms")
                val timestampNs = metadata.getLong("start_timestamp_ns")

                if (timestampMs <= 0 || timestampNs <= 0) {
                    issues.add("Invalid timestamps in metadata")
                }
            }
        } catch (e: Exception) {
            issues.add("Failed to parse metadata: ${e.message}")
        }
    }

    private fun validateRgbData(rgbDir: File, results: MutableMap<String, Boolean>, issues: MutableList<String>) {
        val videoFile = File(rgbDir, "video.mp4")
        val framesDir = File(rgbDir, "frames")
        val csvFile = File(rgbDir, "rgb_frames.csv")

        results["rgb_video"] = videoFile.exists()
        results["rgb_frames_dir"] = framesDir.exists() && framesDir.isDirectory
        results["rgb_csv"] = csvFile.exists()

        if (!videoFile.exists()) issues.add("Missing RGB video file")
        if (!framesDir.exists()) issues.add("Missing RGB frames directory")
        if (!csvFile.exists()) issues.add("Missing RGB CSV file")

        if (csvFile.exists()) {
            validateCsvStructure(csvFile, listOf("timestamp_ns", "timestamp_ms", "frame_number", "filename"), issues)
        }
    }

    private fun validateThermalData(thermalDir: File, results: MutableMap<String, Boolean>, issues: MutableList<String>) {
        val csvFile = File(thermalDir, "thermal_data.csv")
        val imagesDir = File(thermalDir, "thermal_images")

        results["thermal_csv"] = csvFile.exists()
        results["thermal_images"] = imagesDir.exists() && imagesDir.isDirectory

        if (!csvFile.exists()) issues.add("Missing thermal CSV file")

        if (csvFile.exists()) {
            validateCsvStructure(csvFile, listOf("timestamp_ns", "timestamp_ms", "frame_number"), issues)
        }
    }

    private fun validateGsrData(gsrDir: File, results: MutableMap<String, Boolean>, issues: MutableList<String>) {
        val csvFile = File(gsrDir, "gsr.csv")

        results["gsr_csv"] = csvFile.exists()

        if (!csvFile.exists()) issues.add("Missing GSR CSV file")

        if (csvFile.exists()) {
            validateCsvStructure(csvFile, listOf("timestamp_ns", "timestamp_ms"), issues)
        }
    }

    private fun validateAudioData(audioDir: File, results: MutableMap<String, Boolean>, issues: MutableList<String>) {
        val audioFile = audioDir.listFiles { file -> file.name.endsWith(".aac") || file.name.endsWith(".mp4") }?.firstOrNull()
        val eventsFile = File(audioDir, "audio_events.csv")

        results["audio_file"] = audioFile != null
        results["audio_events"] = eventsFile.exists()

        if (audioFile == null) issues.add("Missing audio file")
    }

    private fun validateCsvStructure(csvFile: File, requiredColumns: List<String>, issues: MutableList<String>) {
        try {
            val firstLine = csvFile.bufferedReader().use { it.readLine() }
            if (firstLine != null) {
                val headers = firstLine.split(",").map { it.trim() }
                requiredColumns.forEach { column ->
                    if (!headers.contains(column)) {
                        issues.add("Missing column '$column' in ${csvFile.name}")
                    }
                }
            } else {
                issues.add("Empty CSV file: ${csvFile.name}")
            }
        } catch (e: Exception) {
            issues.add("Failed to validate CSV structure for ${csvFile.name}: ${e.message}")
        }
    }

    private fun validateTimestampConsistency(csvFiles: List<File>, results: MutableMap<String, Boolean>, issues: MutableList<String>) {
        var hasConsistentTimestamps = true

        csvFiles.forEach { csvFile ->
            try {
                csvFile.bufferedReader().use { reader ->
                    val header = reader.readLine()
                    if (header != null) {
                        val hasTimestampNs = header.contains("timestamp_ns")
                        val hasTimestampMs = header.contains("timestamp_ms")

                        if (!hasTimestampNs && !hasTimestampMs) {
                            issues.add("No timestamp columns found in ${csvFile.name}")
                            hasConsistentTimestamps = false
                        }
                    }
                }
            } catch (e: Exception) {
                issues.add("Failed to check timestamps in ${csvFile.name}: ${e.message}")
                hasConsistentTimestamps = false
            }
        }

        results["timestamp_consistency"] = hasConsistentTimestamps
    }

    /**
     * Get storage information for a directory
     */
    fun getStorageInfo(directory: File): StorageInfo {
        return try {
            val statsFs = StatFs(directory.path)
            val totalBytes = statsFs.totalBytes
            val availableBytes = statsFs.availableBytes
            val usedBytes = totalBytes - availableBytes

            StorageInfo(
                totalBytes = totalBytes,
                availableBytes = availableBytes,
                usedBytes = usedBytes,
                usagePercent = (usedBytes.toFloat() / totalBytes * 100f).toInt(),
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get storage info: ${e.message}", e)
            StorageInfo(0, 0, 0, 0)
        }
    }

    data class ValidationResult(
        val sessionId: String,
        val isValid: Boolean,
        val results: Map<String, Boolean>,
        val issues: List<String>,
    )

    data class StorageInfo(
        val totalBytes: Long,
        val availableBytes: Long,
        val usedBytes: Long,
        val usagePercent: Int,
    ) {
        val totalMB: Long get() = totalBytes / (1024 * 1024)
        val availableMB: Long get() = availableBytes / (1024 * 1024)
        val usedMB: Long get() = usedBytes / (1024 * 1024)
    }
}
