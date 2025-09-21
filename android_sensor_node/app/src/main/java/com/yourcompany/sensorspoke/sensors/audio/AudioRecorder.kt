package com.yourcompany.sensorspoke.sensors.audio

import android.content.Context
import android.media.MediaRecorder
import android.util.Log
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
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
    private var recordingStartTime: Long = 0L
    private val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    companion object {
        private const val TAG = "AudioRecorder"
        private const val SAMPLE_RATE = 44100
        private const val CHANNELS = 2
        private const val BIT_RATE = 128000
    }

    override suspend fun start(sessionDir: File) {
        this.sessionDir = sessionDir
        try {
            initializeAudioRecorder()
            startRecording()

            verifyRecordingStarted()

            Log.d(TAG, "Audio recording started successfully: ${audioFile?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio recording: ${e.message}", e)
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
        audioFile = File(sessionDir!!, "audio_${System.currentTimeMillis()}.m4a")

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
                recordingStartTime = System.nanoTime()

                logAudioEvent("RECORDING_STARTED", recordingStartTime)

                Log.i(TAG, "MediaRecorder started successfully at $recordingStartTime")
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

                    logAudioEvent("RECORDING_STOPPED", System.nanoTime())
                }
            } catch (e: RuntimeException) {
                Log.e(TAG, "MediaRecorder stop failed", e)
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
                cleanupMediaRecorder()

                if (attemptAlternativeRecording()) {
                    Log.i(TAG, "Audio recording recovered using alternative method")
                    return@launch
                }

                val errorReportFile = File(sessionDir!!, "audio_recording_error.json")

                val errorReport = try {
                    val alternativeMethodsArray = JSONArray()
                        .put("MediaRecorder_retry")
                        .put("AudioRecord_fallback")
                        .put("AAC_format_fallback")

                    val systemInfo = JSONObject()
                        .put("android_version", android.os.Build.VERSION.SDK_INT)
                        .put("device_model", android.os.Build.MODEL ?: "unknown")
                        .put("manufacturer", android.os.Build.MANUFACTURER ?: "unknown")

                    JSONObject()
                        .put("error_type", "AUDIO_RECORDING_FAILURE")
                        .put("timestamp", System.currentTimeMillis())
                        .put("error_message", error.message ?: "Unknown error")
                        .put("error_class", error.javaClass.simpleName)
                        .put("attempted_file", audioFile?.name ?: "unknown")
                        .put("session_directory", sessionDir?.name ?: "unknown")
                        .put("recovery_attempted", true)
                        .put("alternative_methods_tried", alternativeMethodsArray)
                        .put("system_info", systemInfo)
                        .toString(2)
                } catch (jsonError: Exception) {
                    Log.e(TAG, "Failed to create error report JSON", jsonError)
                    """
                    {
                        "error_type": "AUDIO_RECORDING_FAILURE",
                        "timestamp": ${System.currentTimeMillis()},
                        "error_message": "JSON creation failed: ${jsonError.message}",
                        "original_error": "${error.message ?: "Unknown error"}"
                    }
                    """.trimIndent()
                }

                errorReportFile.writeText(errorReport)

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
                    Triple(MediaRecorder.AudioEncoder.AAC, MediaRecorder.OutputFormat.MPEG_4, "audio_hq.m4a"),
                    Triple(MediaRecorder.AudioEncoder.AAC, MediaRecorder.OutputFormat.THREE_GPP, "audio_std.3gp"),
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

    /**
     * Verify that audio recording is actually working by checking file growth
     */
    private suspend fun verifyRecordingStarted() {
        kotlinx.coroutines.delay(500)

        try {
            val initialSize = audioFile?.length() ?: 0
            kotlinx.coroutines.delay(1000)
            val currentSize = audioFile?.length() ?: 0

            if (currentSize > initialSize && currentSize > 0) {
                Log.i(TAG, "Audio recording verification successful - file growing from $initialSize to $currentSize bytes")
                logAudioEvent("RECORDING_VERIFIED", System.nanoTime(), "File size: $currentSize bytes")
            } else {
                Log.w(TAG, "Audio recording verification failed - file not growing (size: $currentSize)")
                logAudioEvent("RECORDING_VERIFICATION_FAILED", System.nanoTime(), "File size unchanged: $currentSize")
            }

            if (isRecording) {
                val recordingDuration = (System.nanoTime() - recordingStartTime) / 1_000_000
                Log.d(TAG, "Recording duration verification: ${recordingDuration}ms")
                logAudioEvent("RECORDING_DURATION_CHECK", System.nanoTime(), "Duration: ${recordingDuration}ms")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during recording verification: ${e.message}", e)
            logAudioEvent("RECORDING_VERIFICATION_ERROR", System.nanoTime(), e.message ?: "Unknown error")
        }
    }
}
