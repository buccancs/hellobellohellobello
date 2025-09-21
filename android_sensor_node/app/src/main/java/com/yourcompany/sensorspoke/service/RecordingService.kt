package com.yourcompany.sensorspoke.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.IBinder
import android.util.Base64
import android.util.Log
import androidx.core.app.NotificationCompat
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.controller.RecordingController
import com.yourcompany.sensorspoke.network.ConnectionManager
import com.yourcompany.sensorspoke.network.FileTransferManager
import com.yourcompany.sensorspoke.network.NetworkClient
import com.yourcompany.sensorspoke.network.TimeSyncService
import com.yourcompany.sensorspoke.utils.PreviewBus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import java.util.Collections

/**
 * Foreground service that advertises an NSD (Zeroconf) service and hosts a
 * ServerSocket to accept a PC Hub connection and respond to JSON commands.
 *
 * Enhanced to coordinate with the RecordingController for multi-sensor recording.
 * The actual sensor initialization happens in the MainActivity with proper lifecycle management.
 */
class RecordingService : Service() {
    companion object {
        const val ACTION_START_RECORDING = "com.yourcompany.sensorspoke.ACTION_START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.yourcompany.sensorspoke.ACTION_STOP_RECORDING"
        const val ACTION_FLASH_SYNC = "com.yourcompany.sensorspoke.ACTION_FLASH_SYNC"
        const val EXTRA_SESSION_ID = "session_id"
        const val EXTRA_FLASH_TS_NS = "flash_ts_ns"
    }

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var serverSocket: ServerSocket? = null
    private var serverPort: Int = 0
    private lateinit var networkClient: NetworkClient
    private val clientWriters = Collections.synchronizedList(mutableListOf<BufferedWriter>())
    private var previewListener: ((ByteArray, Long) -> Unit)? = null

    private var timeSyncService: TimeSyncService? = null
    private var connectionManager: ConnectionManager? = null

    private var recordingController: RecordingController? = null

    private var currentSessionId: String? = null
    private var isRecording: Boolean = false

    override fun onCreate() {
        super.onCreate()
        networkClient = NetworkClient(applicationContext)

        initializeRecordingSystem()

        timeSyncService = TimeSyncService(applicationContext)
        connectionManager = ConnectionManager(applicationContext, networkClient).apply {
            onConnectionEstablished = { address, port ->
                timeSyncService?.startSync(address, 8081)
            }
            onConnectionLost = {
                timeSyncService?.stopSync()
            }
        }

        startInForeground()
        previewListener = { bytes, ts -> broadcastPreviewFrame(bytes, ts) }
        previewListener?.let { PreviewBus.subscribe(it) }
        scope.launch { startServerAndAdvertise() }
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            networkClient.unregister()
        } catch (_: Exception) {
        }
        try {
            serverSocket?.close()
        } catch (_: Exception) {
        }

        timeSyncService?.cleanup()
        connectionManager?.cleanup()

        cleanupRecordingSystem()

        previewListener?.let { PreviewBus.unsubscribe(it) }
        previewListener = null
        scope.cancel()
    }

    /**
     * Initialize the recording controller and sensor recorders
     */
    private fun initializeRecordingSystem() {
        try {
            recordingController = RecordingController(applicationContext)


            Log.i("RecordingService", "Recording system initialized for network coordination")
        } catch (e: Exception) {
            Log.e("RecordingService", "Failed to initialize recording system: ${e.message}", e)
        }
    }

    /**
     * Cleanup the recording system
     */
    private fun cleanupRecordingSystem() {
        try {
            recordingController = null

            Log.i("RecordingService", "Recording system cleaned up")
        } catch (e: Exception) {
            Log.e("RecordingService", "Error during recording system cleanup: ${e.message}", e)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startInForeground() {
        val channelId = "recording_service_channel"
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel =
                NotificationChannel(
                    channelId,
                    getString(R.string.notification_channel_name),
                    NotificationManager.IMPORTANCE_LOW,
                )
            channel.description = getString(R.string.notification_channel_desc)
            nm.createNotificationChannel(channel)
        }
        val notification: Notification =
            NotificationCompat
                .Builder(this, channelId)
                .setContentTitle(getString(R.string.notification_title))
                .setContentText(getString(R.string.notification_text))
                .setSmallIcon(R.drawable.ic_notification_recording)
                .setOngoing(true)
                .build()
        startForeground(1001, notification)
    }

    private suspend fun startServerAndAdvertise() {
        serverSocket = ServerSocket(0)
        serverPort = serverSocket!!.localPort
        networkClient.register("_gsr-controller._tcp.", "SensorSpoke - ${Build.MODEL}", serverPort)

        while (scope.isActive) {
            try {
                val socket = serverSocket!!.accept()
                scope.launch { handleConnection(socket) }
            } catch (_: Exception) {
                break
            }
        }
    }

    private fun handleConnection(socket: Socket) {
        socket.use { s ->
            val bis = BufferedInputStream(s.getInputStream(), 8192)
            val writer = BufferedWriter(OutputStreamWriter(s.getOutputStream()))
            clientWriters.add(writer)
            maybeSendRejoin(writer)
            try {
                while (true) {
                    val text = readJsonFromSocket(bis) ?: break
                    val obj = JSONObject(text)
                    val isV1 = obj.optInt("v", 0) == 1
                    val command = obj.optString("command")
                    val id = obj.optInt("id", 0)
                    when (command) {
                        "query_capabilities" -> {
                            val caps = collectCapabilities()
                            val response =
                                JSONObject()
                                    .put("ack_id", id)
                                    .put("status", "ok")
                                    .put("capabilities", caps)
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }

                        "time_sync" -> {
                            val t1 = System.nanoTime()
                            val t2 = System.nanoTime()
                            val response =
                                JSONObject()
                                    .put("ack_id", id)
                                    .put("status", "ok")
                                    .put("t1", t1)
                                    .put("t2", t2)
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }

                        "start_recording" -> {
                            val sessionId = obj.optString("session_id", "")

                            val intent = Intent(ACTION_START_RECORDING)
                                .putExtra(EXTRA_SESSION_ID, sessionId)
                                .setPackage("com.yourcompany.sensorspoke")
                            sendBroadcast(intent)

                            if (sessionId.isNotEmpty()) currentSessionId = sessionId
                            isRecording = true

                            updateNotificationForRecording(currentSessionId)

                            val response = JSONObject().put("ack_id", id).put("status", "ok")
                                .put("session_id", sessionId)
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }

                            Log.i("RecordingService", "Recording start command sent for session: $sessionId")
                        }

                        "stop_recording" -> {
                            val intent = Intent(ACTION_STOP_RECORDING)
                                .setPackage("com.yourcompany.sensorspoke")
                            sendBroadcast(intent)

                            isRecording = false

                            updateNotificationForIdle()

                            val response = JSONObject().put("ack_id", id).put("status", "ok")
                                .put("session_id", currentSessionId ?: "")
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }

                            Log.i("RecordingService", "Recording stop command sent for session: $currentSessionId")
                        }

                        "flash_sync" -> {
                            val ts = System.nanoTime()
                            val intent =
                                Intent(ACTION_FLASH_SYNC)
                                    .putExtra(EXTRA_FLASH_TS_NS, ts)
                                    .setPackage("com.yourcompany.sensorspoke")
                            sendBroadcast(intent)
                            val response = JSONObject().put("ack_id", id).put("status", "ok").put("ts", ts)
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }

                        "transfer_files" -> {
                            val host = obj.optString("host", "")
                            val port = obj.optInt("port", -1)
                            val sessionId = obj.optString("session_id", "")
                            if (host.isNotEmpty() && port > 0 && sessionId.isNotEmpty()) {
                                scope.launch {
                                    runCatching {
                                        val ftm = FileTransferManager(applicationContext)
                                        ftm.transferSession(sessionId, host, port)
                                    }
                                }
                                val response = JSONObject().put("ack_id", id).put("status", "ok")
                                if (isV1) {
                                    response.put("v", 1).put("type", "ack")
                                    safeWriteFrame(writer, response)
                                } else {
                                    safeWriteLine(writer, response.toString())
                                }
                            } else {
                                val response =
                                    JSONObject()
                                        .put("ack_id", id)
                                        .put("status", "error")
                                        .put("message", "invalid parameters")
                                if (isV1) {
                                    response.put("v", 1).put("type", "error").put("code", "E_BAD_PARAM")
                                    safeWriteFrame(writer, response)
                                } else {
                                    safeWriteLine(writer, response.toString())
                                }
                            }
                        }

                        else -> {
                            val response =
                                JSONObject()
                                    .put("ack_id", id)
                                    .put("status", "error")
                                    .put("message", "unknown command")
                            if (isV1) {
                                response.put("v", 1).put("type", "error").put("code", "E_UNKNOWN_CMD")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }
                    }
                }
            } catch (_: Exception) {
            } finally {
                try {
                    writer.flush()
                } catch (_: Exception) {
                }
                clientWriters.remove(writer)
            }
        }
    }

    private fun safeWriteLine(
        writer: BufferedWriter,
        text: String,
    ) {
        try {
            synchronized(writer) {
                writer.write(text)
                writer.write("\n")
                writer.flush()
            }
        } catch (_: Exception) {
        }
    }

    private fun safeWriteFrame(
        writer: BufferedWriter,
        obj: JSONObject,
    ) {
        try {
            synchronized(writer) {
                val json = obj.toString()
                val len = json.toByteArray(Charsets.UTF_8).size
                writer.write(len.toString())
                writer.write("\n")
                writer.write(json)
                writer.flush()
            }
        } catch (_: Exception) {
        }
    }

    private fun maybeSendRejoin(writer: BufferedWriter) {
        val sid = currentSessionId
        if (sid.isNullOrEmpty()) return
        try {
            val obj =
                JSONObject()
                    .put("v", 1)
                    .put("type", "cmd")
                    .put("command", "rejoin_session")
                    .put("session_id", sid)
                    .put("device_id", Build.MODEL ?: "device")
                    .put("recording", isRecording)
            safeWriteFrame(writer, obj)
        } catch (_: Exception) {
        }
    }

    private fun readJsonFromSocket(bis: BufferedInputStream): String? {
        val first = readAsciiLine(bis) ?: return null
        val isDigits = first.isNotEmpty() && first.all { it in '0'..'9' }
        return if (isDigits) {
            val length =
                try {
                    first.toInt()
                } catch (_: Exception) {
                    return null
                }
            val buf = ByteArray(length)
            var off = 0
            while (off < length) {
                val r = bis.read(buf, off, length - off)
                if (r == -1) return null
                off += r
            }
            String(buf, Charsets.UTF_8)
        } else {
            first
        }
    }

    private fun readAsciiLine(bis: BufferedInputStream): String? {
        val out = ByteArrayOutputStream()
        while (true) {
            val b = bis.read()
            if (b == -1) break
            if (b == '\n'.code) break
            if (b != '\r'.code) out.write(b)
        }
        if (out.size() == 0) return null
        return out.toString("US-ASCII")
    }

    private fun broadcastPreviewFrame(
        bytes: ByteArray,
        tsNs: Long,
    ) {
        val b64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
        val obj =
            JSONObject()
                .put("v", 1)
                .put("type", "event")
                .put("name", "preview_frame")
                .put("device_id", Build.MODEL ?: "device")
                .put("jpeg_base64", b64)
                .put("ts", tsNs)
        val snapshot: List<BufferedWriter> = synchronized(clientWriters) { clientWriters.toList() }
        for (w in snapshot) {
            safeWriteFrame(w, obj)
        }
    }

    private fun collectCapabilities(): JSONObject {
        val result = JSONObject()
        val model = Build.MODEL ?: "unknown"
        result.put("device_id", model)
        result.put("device_model", model)
        result.put("android_sdk", Build.VERSION.SDK_INT)
        result.put("android_release", Build.VERSION.RELEASE ?: "")
        result.put("service_port", serverPort)

        val controller = recordingController
        if (controller != null) {
            val sensorStatus = controller.getSensorStatusReport()
            result.put("has_rgb", sensorStatus.containsKey("rgb"))
            result.put("has_thermal", sensorStatus.containsKey("thermal"))
            result.put("has_gsr", sensorStatus.containsKey("gsr"))
            result.put("sensor_status", JSONObject(sensorStatus))
        } else {
            result.put("has_rgb", true)
            result.put("has_thermal", true)
            result.put("has_gsr", true)
        }

        val cameras = JSONArray()
        try {
            val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            for (id in cm.cameraIdList) {
                val chars = cm.getCameraCharacteristics(id)
                val facingInt = chars.get(CameraCharacteristics.LENS_FACING)
                val facing =
                    when (facingInt) {
                        CameraCharacteristics.LENS_FACING_FRONT -> "front"
                        CameraCharacteristics.LENS_FACING_BACK -> "back"
                        CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                        else -> "unknown"
                    }
                cameras.put(JSONObject().put("id", id).put("facing", facing))
            }
        } catch (_: Exception) {
        }
        result.put("cameras", cameras)
        return result
    }

    /**
     * Update notification to show recording status
     */
    private fun updateNotificationForRecording(sessionId: String?) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "recording_service_channel"

        val notification: Notification =
            NotificationCompat
                .Builder(this, channelId)
                .setContentTitle("Recording in Progress")
                .setContentText("Session: ${sessionId ?: "Unknown"}")
                .setSmallIcon(R.drawable.ic_notification_recording)
                .setOngoing(true)
                .setProgress(0, 0, true)
                .build()

        nm.notify(1001, notification)
    }

    /**
     * Update notification to show idle status
     */
    private fun updateNotificationForIdle() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channelId = "recording_service_channel"

        val notification: Notification =
            NotificationCompat
                .Builder(this, channelId)
                .setContentTitle(getString(R.string.notification_title))
                .setContentText(getString(R.string.notification_text))
                .setSmallIcon(R.drawable.ic_notification_recording)
                .setOngoing(true)
                .build()

        nm.notify(1001, notification)
    }
}
