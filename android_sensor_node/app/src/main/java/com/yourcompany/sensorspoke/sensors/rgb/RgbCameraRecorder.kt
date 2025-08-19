package com.yourcompany.sensorspoke.sensors.rgb

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import com.yourcompany.sensorspoke.utils.EmulatorUtils
import com.yourcompany.sensorspoke.utils.PreviewBus
import com.yourcompany.sensorspoke.utils.TimeManager
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * RgbCameraRecorder using CameraX to record a 1080p MP4 and capture high-res JPEGs with
 * nanosecond timestamps in filenames.
 */
class RgbCameraRecorder(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
) : SensorRecorder {
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var lastPreviewNs: Long = 0L
    private var csvWriter: java.io.BufferedWriter? = null
    private var csvFile: File? = null

    override suspend fun start(sessionDir: File) {
        // Ensure directories
        val rgbDir = sessionDir
        if (!rgbDir.exists()) rgbDir.mkdirs()
        val framesDir = File(rgbDir, "frames").apply { mkdirs() }
        
        // Open CSV for JPEG index
        csvFile = File(rgbDir, "rgb.csv")
        csvWriter = java.io.BufferedWriter(java.io.FileWriter(csvFile!!, true))
        if (csvFile!!.length() == 0L) {
            csvWriter!!.write("timestamp_ns,filename\n")
            csvWriter!!.flush()
        }
        
        // Check if running on emulator and if camera is available
        val isEmulator = EmulatorUtils.isRunningOnEmulator()
        val hasCamera = EmulatorUtils.isCameraAvailable(context)
        
        if (isEmulator && !hasCamera) {
            android.util.Log.i("RgbCameraRecorder", "Running on emulator without camera - creating simulation mode files")
            // Create placeholder video file for emulator
            val sessionStartTs = TimeManager.nowNanos()
            val videoFile = File(rgbDir, "video_${sessionStartTs}_emulator_placeholder.mp4")
            videoFile.createNewFile()
            return
        }
        
        try {
            // Use session start timestamp in video filename for traceability
            val sessionStartTs = TimeManager.nowNanos()
            val videoFile = File(rgbDir, "video_$sessionStartTs.mp4")

            val provider = ProcessCameraProvider.getInstance(context).get()
            cameraProvider = provider

            // Build Recorder for 1080p
            val recorder =
                Recorder.Builder()
                    .setQualitySelector(
                        QualitySelector.from(
                            Quality.FHD,
                            FallbackStrategy.higherQualityOrLowerThan(Quality.FHD),
                        ),
                    )
                    .build()
            videoCapture = VideoCapture.withOutput(recorder)

            // ImageCapture for high-res stills
            imageCapture =
                ImageCapture.Builder()
                    .setTargetRotation(Surface.ROTATION_0)
                    .setJpegQuality(95)
                    .build()

                // Unbind then bind
            provider.unbindAll()
            val useCases = mutableListOf<androidx.camera.core.UseCase>()
            imageCapture?.let { useCases.add(it) }
            videoCapture?.let { useCases.add(it) }
            provider.bindToLifecycle(lifecycleOwner, cameraSelector, *useCases.toTypedArray())

            // Start video recording
            val outputOpts = FileOutputOptions.Builder(videoFile).build()
            recording =
                videoCapture!!.output
                    .prepareRecording(context, outputOpts)
                    .start(ContextCompat.getMainExecutor(context)) { /* events ignored */ }

            // Start still capture loop throttled to ~6–8 FPS for preview + file archival
            scope.launch {
                while (isActive) {
                    val ts = TimeManager.nowNanos()
                    val outFile = File(framesDir, "frame_$ts.jpg")
                    val output = ImageCapture.OutputFileOptions.Builder(outFile).build()
                val ic = imageCapture ?: break
                ic.takePicture(
                    output,
                    executor,
                    object : ImageCapture.OnImageSavedCallback {
                        override fun onError(exception: ImageCaptureException) {
                            // Keep loop running on errors
                        }

                        override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                            // Log JPEG to CSV: timestamp_ns,filename (relative)
                            try {
                                csvWriter?.apply {
                                    write("$ts,frames/${outFile.name}\n")
                                    flush()
                                }
                            } catch (_: Exception) {
                            }
                            // Build a downsampled, low-quality JPEG preview and emit
                            val now = TimeManager.nowNanos()
                            // throttle to ~6–8 FPS
                            if (now - lastPreviewNs >= 150_000_000L) {
                                lastPreviewNs = now
                                runCatching {
                                    val bmp = BitmapFactory.decodeFile(outFile.absolutePath) ?: return@runCatching
                                    val w = 320
                                    val h = 240
                                    val scaled = Bitmap.createScaledBitmap(bmp, w, h, true)
                                    val baos = ByteArrayOutputStream()
                                    scaled.compress(Bitmap.CompressFormat.JPEG, 40, baos)
                                    val bytes = baos.toByteArray()
                                    PreviewBus.emit(bytes, now)
                                    try {
                                        baos.close()
                                    } catch (_: Exception) {
                                    }
                                    if (scaled != bmp) {
                                        bmp.recycle()
                                    }
                                    scaled.recycle()
                                }
                            }
                        }
                    },
                )
                // ~6–8 FPS loop (~150ms)
                delay(150)
            }
        }
        
        } catch (e: Exception) {
            android.util.Log.e("RgbCameraRecorder", "Camera initialization failed", e)
            // Create placeholder files for graceful degradation
            val sessionStartTs = TimeManager.nowNanos()
            val videoFile = File(rgbDir, "video_${sessionStartTs}_error.mp4")
            videoFile.createNewFile()
            throw e
        }
    }

    override suspend fun stop() {
        try {
            recording?.stop()
        } catch (_: Exception) {
        }
        recording = null
        try {
            cameraProvider?.unbindAll()
        } catch (_: Exception) {
        }
        cameraProvider = null
        imageCapture = null
        videoCapture = null
        // Cancel capture loop before closing CSV to avoid writes after close
        scope.coroutineContext[Job]?.cancel()
        // Close CSV resources
        try {
            csvWriter?.flush()
        } catch (_: Exception) {
        }
        try {
            csvWriter?.close()
        } catch (_: Exception) {
        }
        csvWriter = null
        csvFile = null
        executor.shutdown()
    }
}
