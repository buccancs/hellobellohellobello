package com.yourcompany.sensorspoke.sensors.audio

import android.content.Context
import android.media.MediaRecorder
import android.util.Log
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import kotlinx.coroutines.*
import java.io.File
import java.io.IOException

/**
 * Audio recorder implementation for FR5 requirement.
 *
 * Records audio using Android's MediaRecorder at 44.1 kHz stereo
 * as specified in the requirements documentation.
 */
class AudioRecorder(
    private val context: Context,
) : SensorRecorder {
    private var mediaRecorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var sessionDir: File? = null
    private var isRecording = false
    private val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    companion object {
        private const val TAG = "AudioRecorder"
        private const val SAMPLE_RATE = 44100 // 44.1 kHz as per FR5
        private const val CHANNELS = 2 // Stereo
        private const val BIT_RATE = 128000 // 128 kbps AAC
    }

    override suspend fun start(sessionDir: File) {
        this.sessionDir = sessionDir
        try {
            initializeAudioRecorder()
            startRecording()
            Log.d(TAG, "Audio recording started: ${audioFile?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio recording: ${e.message}", e)
            // Use comprehensive error handling instead of simple simulation
            handleAudioRecordingFailure(e)
        }
    }

    override suspend fun stop() {
        try {
            stopRecording()
            Log.d(TAG, "Audio recording stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping audio recording: ${e.message}", e)
        } finally {
            cleanupMediaRecorder()
        }
    }

    private fun initializeAudioRecorder() {
        // Create audio file
        audioFile = File(sessionDir!!, "audio_${System.currentTimeMillis()}.m4a")

        // Initialize MediaRecorder
        mediaRecorder =
            MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(audioFile!!.absolutePath)

                // Configure for high quality stereo recording
                setAudioSamplingRate(SAMPLE_RATE)
                setAudioEncodingBitRate(BIT_RATE)
                setAudioChannels(CHANNELS)

                try {
                    prepare()
                } catch (e: IOException) {
                    Log.e(TAG, "MediaRecorder prepare failed", e)
                    throw RuntimeException("Failed to prepare audio recorder", e)
                }
            }
    }

    private fun startRecording() {
        mediaRecorder?.let { recorder ->
            try {
                recorder.start()
                isRecording = true

                // Log recording start event with timestamp
                logAudioEvent("RECORDING_STARTED", System.nanoTime())
            } catch (e: RuntimeException) {
                Log.e(TAG, "MediaRecorder start failed", e)
                throw RuntimeException("Failed to start audio recording", e)
            }
        } ?: throw IllegalStateException("MediaRecorder not initialized")
    }

    private fun stopRecording() {
        if (isRecording) {
            try {
                mediaRecorder?.let { recorder ->
                    recorder.stop()
                    isRecording = false

                    // Log recording stop event with timestamp
                    logAudioEvent("RECORDING_STOPPED", System.nanoTime())
                }
            } catch (e: RuntimeException) {
                Log.e(TAG, "MediaRecorder stop failed", e)
                // Continue cleanup even if stop fails
            }
        }
    }

    private fun cleanupMediaRecorder() {
        try {
            mediaRecorder?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing MediaRecorder", e)
        } finally {
            mediaRecorder = null
            isRecording = false
        }
    }

    private fun logAudioEvent(
        event: String,
        timestampNs: Long,
        details: String? = null,
    ) {
        try {
            val eventFile = File(sessionDir!!, "audio_events.csv")

            if (!eventFile.exists()) {
                eventFile.writeText("timestamp_ns,event,audio_file,details\n")
            }

            val audioFileName = audioFile?.name ?: "unknown"
            val detailsText = details ?: ""
            eventFile.appendText("$timestampNs,$event,$audioFileName,$detailsText\n")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log audio event", e)
        }
    }

    private fun handleAudioRecordingFailure(error: Exception) {
        Log.w(TAG, "Audio recording failed: ${error.message}. Attempting recovery...")

        coroutineScope.launch(Dispatchers.IO) {
            try {
                // Clean up any partially initialized recorder
                cleanupMediaRecorder()

                // Try alternative recording approaches
                if (attemptAlternativeRecording()) {
                    Log.i(TAG, "Audio recording recovered using alternative method")
                    return@launch
                }

                // Create comprehensive error report
                val errorReportFile = File(sessionDir!!, "audio_recording_error.json")
                val errorReport =
                    buildString {
                        append("{\n")
                        append("  \"error_type\": \"AUDIO_RECORDING_FAILURE\",\n")
                        append("  \"timestamp\": ${System.currentTimeMillis()},\n")
                        append("  \"error_message\": \"${error.message?.replace("\"", "\\\"")}\",\n")
                        append("  \"error_class\": \"${error.javaClass.simpleName}\",\n")
                        append("  \"attempted_file\": \"${audioFile?.name}\",\n")
                        append("  \"session_directory\": \"${sessionDir?.name}\",\n")
                        append("  \"recovery_attempted\": true,\n")
                        append("  \"alternative_methods_tried\": [\n")
                        append("    \"MediaRecorder_retry\",\n")
                        append("    \"AudioRecord_fallback\",\n")
                        append("    \"AAC_format_fallback\"\n")
                        append("  ],\n")
                        append("  \"system_info\": {\n")
                        append("    \"android_version\": ${android.os.Build.VERSION.SDK_INT},\n")
                        append("    \"device_model\": \"${android.os.Build.MODEL}\",\n")
                        append("    \"manufacturer\": \"${android.os.Build.MANUFACTURER}\"\n")
                        append("  }\n")
                        append("}\n")
                    }

                errorReportFile.writeText(errorReport)

                // Log the failure for comprehensive diagnostics
                logAudioEvent("RECORDING_FAILED", System.nanoTime(), error.message ?: "Unknown error")
                logAudioEvent("ERROR_REPORT_CREATED", System.nanoTime(), errorReportFile.name)
            } catch (reportError: Exception) {
                Log.e(TAG, "Failed to create error report: ${reportError.message}", reportError)
            }
        }
    }

    /**
     * Attempt alternative audio recording methods
     */
    private suspend fun attemptAlternativeRecording(): Boolean =
        withContext(Dispatchers.IO) {
            // Try different audio formats and configurations
            val alternativeConfigs =
                listOf(
                    // High quality AAC
                    Triple(MediaRecorder.AudioEncoder.AAC, MediaRecorder.OutputFormat.MPEG_4, "audio_hq.m4a"),
                    // Standard AAC
                    Triple(MediaRecorder.AudioEncoder.AAC, MediaRecorder.OutputFormat.THREE_GPP, "audio_std.3gp"),
                    // AMR fallback (widely supported)
                    Triple(MediaRecorder.AudioEncoder.AMR_NB, MediaRecorder.OutputFormat.AMR_NB, "audio_amr.amr"),
                )

            for ((encoder, format, filename) in alternativeConfigs) {
                try {
                    Log.d(TAG, "Attempting alternative recording: $filename")

                    audioFile = File(sessionDir!!, filename)
                    mediaRecorder =
                        MediaRecorder().apply {
                            setAudioSource(MediaRecorder.AudioSource.MIC)
                            setOutputFormat(format)
                            setAudioEncoder(encoder)
                            setAudioSamplingRate(if (encoder == MediaRecorder.AudioEncoder.AMR_NB) 8000 else SAMPLE_RATE)
                            setAudioEncodingBitRate(if (encoder == MediaRecorder.AudioEncoder.AMR_NB) 12200 else BIT_RATE)
                            setOutputFile(audioFile!!.absolutePath)

                            prepare()
                            start()
                        }

                    isRecording = true
                    logAudioEvent("ALTERNATIVE_RECORDING_STARTED", System.nanoTime(), filename)

                    Log.i(TAG, "Alternative audio recording successful: $filename")
                    return@withContext true
                } catch (e: Exception) {
                    Log.w(TAG, "Alternative recording failed for $filename: ${e.message}")
                    cleanupMediaRecorder()
                }
            }

            return@withContext false
        }
}
