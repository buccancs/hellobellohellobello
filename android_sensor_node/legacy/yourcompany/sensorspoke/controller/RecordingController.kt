package com.yourcompany.sensorspoke.controller

import android.content.Context
import android.os.Build
import com.yourcompany.sensorspoke.sensors.SensorRecorder
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * RecordingController coordinates start/stop across sensor recorders and manages
 * the lifecycle of a recording session, including session directory creation.
 */
class RecordingController(
    private val context: Context?,
    private val sessionsRootOverride: File? = null,
) {
    data class RecorderEntry(
        val name: String,
        val recorder: SensorRecorder,
    )

    enum class State { IDLE, PREPARING, RECORDING, STOPPING }

    private val _state = MutableStateFlow(State.IDLE)
    val state: StateFlow<State> = _state

    private val _currentSessionId = MutableStateFlow<String?>(null)
    val currentSessionId: StateFlow<String?> = _currentSessionId

    private val recorders = mutableListOf<RecorderEntry>()
    private var sessionRootDir: File? = null

    /**
     * Register a recorder under a unique name; its data will be stored under sessionDir/<name>.
     */
    fun register(
        name: String,
        recorder: SensorRecorder,
    ) {
        require(recorders.none { it.name == name }) { "Recorder name '$name' already registered" }
        recorders.add(RecorderEntry(name, recorder))
    }

    /**
     * Starts a new session with an optional provided sessionId. When null, a new id is generated.
     * Creates the session directory tree and starts all registered recorders.
     */
    suspend fun startSession(sessionId: String? = null) {
        if (_state.value != State.IDLE) return
        _state.value = State.PREPARING
        try {
            val id = sessionId ?: generateSessionId()
            val root = ensureSessionsRoot()
            val sessionDir = File(root, id)
            if (!sessionDir.exists()) sessionDir.mkdirs()
            // Create per-recorder subdirs first
            for (entry in recorders) {
                File(sessionDir, entry.name).mkdirs()
            }
            // Start all recorders
            for (entry in recorders) {
                val sub = File(sessionDir, entry.name)
                entry.recorder.start(sub)
            }
            sessionRootDir = sessionDir
            _currentSessionId.value = id
            _state.value = State.RECORDING
        } catch (t: Throwable) {
            // Best-effort cleanup: stop any started recorders
            safeStopAll()
            _currentSessionId.value = null
            sessionRootDir = null
            _state.value = State.IDLE
            throw t
        }
    }

    /**
     * Stops the current session and all recorders, leaving files on disk.
     */
    suspend fun stopSession() {
        if (_state.value != State.RECORDING) return
        _state.value = State.STOPPING
        try {
            for (entry in recorders) {
                runCatching { entry.recorder.stop() }
            }
        } finally {
            _state.value = State.IDLE
            _currentSessionId.value = null
            sessionRootDir = null
        }
    }

    private fun ensureSessionsRoot(): File {
        sessionsRootOverride?.let {
            if (!it.exists()) it.mkdirs()
            return it
        }
        val ctx = context
        val root: File =
            if (ctx != null) {
                val base: File? = ctx.getExternalFilesDir(null)
                File(base ?: ctx.filesDir, "sessions")
            } else {
                File(System.getProperty("java.io.tmpdir"), "sessions")
            }
        if (!root.exists()) root.mkdirs()
        return root
    }

    private fun generateSessionId(): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
        val ts = sdf.format(Date())
        val shortUuid = UUID.randomUUID().toString().substring(0, 8)
        val model = Build.MODEL?.replace(" ", "_") ?: "device"
        return "${ts}_${model}_$shortUuid"
    }

    private suspend fun safeStopAll() {
        for (entry in recorders) {
            runCatching { entry.recorder.stop() }
        }
    }
}
