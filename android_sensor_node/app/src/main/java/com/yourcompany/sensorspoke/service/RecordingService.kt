package com.yourcompany.sensorspoke.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.IBinder
import android.util.Base64
import android.util.Log
import androidx.core.app.NotificationCompat
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.network.FileTransferManager
import com.yourcompany.sensorspoke.network.NetworkClient
import com.yourcompany.sensorspoke.utils.EmulatorUtils
import com.yourcompany.sensorspoke.utils.PreviewBus
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import java.util.*

/**
 * Foreground service that advertises an NSD (Zeroconf) service and hosts a
 * ServerSocket to accept a PC Hub connection and respond to JSON commands.
 *
 * Phase 1 implements only the query_capabilities handshake.
 */
class RecordingService : Service() {
    companion object {
        const val ACTION_START_RECORDING = "com.yourcompany.sensorspoke.ACTION_START_RECORDING"
        const val ACTION_STOP_RECORDING = "com.yourcompany.sensorspoke.ACTION_STOP_RECORDING"
        const val ACTION_FLASH_SYNC = "com.yourcompany.sensorspoke.ACTION_FLASH_SYNC"
        const val EXTRA_SESSION_ID = "session_id"
        const val EXTRA_FLASH_TS_NS = "flash_ts_ns"
    }

    private val scope = CoroutineScope(Dispatchers.IO + Job())
    private var serverSocket: ServerSocket? = null
    private var serverPort: Int = 0
    private lateinit var networkClient: NetworkClient
    private val clientWriters = Collections.synchronizedList(mutableListOf<BufferedWriter>())
    private var previewListener: ((ByteArray, Long) -> Unit)? = null

    // FR8: track current session state to support rejoin notification
    private var currentSessionId: String? = null
    private var isRecording: Boolean = false

    override fun onCreate() {
        super.onCreate()
        networkClient = NetworkClient(applicationContext)
        startInForeground()
        // Subscribe to preview frames and forward to connected clients
        previewListener = { bytes, ts -> broadcastPreviewFrame(bytes, ts) }
        previewListener?.let { PreviewBus.subscribe(it) }
        scope.launch { startServerAndAdvertise() }
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        // Keep running until explicitly stopped
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
        previewListener?.let { PreviewBus.unsubscribe(it) }
        previewListener = null
        scope.cancel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startInForeground() {
        try {
            val channelId = "recording_service_channel"
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            
            Log.i("RecordingService", "Starting foreground service - ${EmulatorUtils.getEnvironmentInfo()}")
            
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
                NotificationCompat.Builder(this, channelId)
                    .setContentTitle(getString(R.string.notification_title))
                    .setContentText(getString(R.string.notification_text))
                    .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth) // placeholder icon
                    .setOngoing(true)
                    .build()
            startForeground(1001, notification)
            
        } catch (e: Exception) {
            Log.e("RecordingService", "Failed to start foreground service", e)
            // Continue running as background service if foreground fails
        }
    }

    private suspend fun startServerAndAdvertise() {
        serverSocket = ServerSocket(0) // auto-assign free port
        serverPort = serverSocket!!.localPort
        networkClient.register("_gsr-controller._tcp.", "SensorSpoke - ${Build.MODEL}", serverPort)

        while (scope.isActive) {
            try {
                val socket = serverSocket!!.accept()
                scope.launch { handleConnection(socket) }
            } catch (_: Exception) {
                // Socket closed or error; exit loop on service stop
                break
            }
        }
    }

    private fun handleConnection(socket: Socket) {
        socket.use { s ->
            val bis = BufferedInputStream(s.getInputStream(), 8192)
            val writer = BufferedWriter(OutputStreamWriter(s.getOutputStream()))
            // Track this client for preview broadcasting
            clientWriters.add(writer)
            // FR8: On new connection, notify PC of current or last session
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
                            // PC sent t0; record t1 on arrival and reply immediately with t1 and t2
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
                            // Forward to UI via broadcast
                            val intent =
                                Intent(ACTION_START_RECORDING)
                                    .putExtra(EXTRA_SESSION_ID, sessionId)
                                    .setPackage("com.yourcompany.sensorspoke") // Fix lint: UnsafeImplicitIntentLaunch
                            sendBroadcast(intent)
                            // FR8: update local session state for rejoin purposes
                            if (sessionId.isNotEmpty()) currentSessionId = sessionId
                            isRecording = true
                            val response = JSONObject().put("ack_id", id).put("status", "ok")
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }

                        "stop_recording" -> {
                            val intent =
                                Intent(ACTION_STOP_RECORDING)
                                    .setPackage("com.yourcompany.sensorspoke") // Fix lint: UnsafeImplicitIntentLaunch
                            sendBroadcast(intent)
                            // FR8: update local session state — session ended but keep id for rejoin-triggered transfer
                            isRecording = false
                            val response = JSONObject().put("ack_id", id).put("status", "ok")
                            if (isV1) {
                                response.put("v", 1).put("type", "ack")
                                safeWriteFrame(writer, response)
                            } else {
                                safeWriteLine(writer, response.toString())
                            }
                        }

                        "flash_sync" -> {
                            val ts = System.nanoTime()
                            val intent =
                                Intent(ACTION_FLASH_SYNC)
                                    .putExtra(EXTRA_FLASH_TS_NS, ts)
                                    .setPackage("com.yourcompany.sensorspoke") // Fix lint: UnsafeImplicitIntentLaunch
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
                                // Start transfer asynchronously
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
                            // Unknown commands acknowledged as error
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
                // swallow errors but keep service alive
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
            // ignore write errors; connection handler will close
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
            // ignore write errors; connection handler will close
        }
    }

    // FR8 helper: notify PC of current or last session upon connection
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
        // Read first line
        val first = readAsciiLine(bis) ?: return null
        val isDigits = first.isNotEmpty() && first.all { it in '0'..'9' }
        return if (isDigits) {
            // length-prefixed payload follows
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
            // legacy newline-delimited JSON
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
        // Sensor availability (software capability flags)
        result.put("has_rgb", true)
        result.put("has_thermal", true)
        result.put("has_gsr", true)
        val cameras = JSONArray()
        try {
            val cm = getSystemService(CAMERA_SERVICE) as CameraManager
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
            // ignore camera errors
        }
        result.put("cameras", cameras)
        return result
    }
}
