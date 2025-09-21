package com.yourcompany.sensorspoke.network

import android.content.Context
import android.os.Build
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Phase 3: Enhanced Protocol v2.0 for Advanced Networking Features
 *
 * Extends the basic JSON protocol with advanced session management,
 * flash synchronization, status updates, and enhanced error handling.
 */
object EnhancedProtocol {
    private const val TAG = "EnhancedProtocol"
    const val PROTOCOL_VERSION = "2.0"

    /**
     * Message types for enhanced protocol
     */
    object MessageTypes {
        const val QUERY_CAPABILITIES = "query_capabilities"
        const val START_RECORDING = "start_recording"
        const val STOP_RECORDING = "stop_recording"
        const val FLASH_SYNC = "flash_sync"

        const val SESSION_START = "session_start"
        const val SESSION_STOP = "session_stop"
        const val SESSION_STATUS = "session_status"
        const val SYNC_FLASH = "sync_flash"
        const val STATUS_UPDATE = "status_update"
        const val HEARTBEAT = "heartbeat"
        const val DEVICE_INFO = "device_info"
        const val TIME_SYNC_REQUEST = "time_sync_request"
        const val TIME_SYNC_RESPONSE = "time_sync_response"
        const val ERROR_REPORT = "error_report"
        const val RECONNECT_REQUEST = "reconnect_request"
    }

    /**
     * Create enhanced device capabilities response
     */
    fun createCapabilitiesResponse(): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "response")
            put("command", MessageTypes.QUERY_CAPABILITIES)
            put("status", "ok")
            put("protocol_version", PROTOCOL_VERSION)
            put("device_info", createDeviceInfo())
            put("capabilities", createCapabilitiesArray())
            put("features", createFeaturesArray())
        }
    }

    /**
     * Create comprehensive device information
     */
    fun createDeviceInfo(): JSONObject {
        return JSONObject().apply {
            put("device_id", "${Build.MANUFACTURER}_${Build.MODEL}_${System.currentTimeMillis()}")
            put("device_name", "${Build.MANUFACTURER} ${Build.MODEL}")
            put("android_version", Build.VERSION.RELEASE)
            put("api_level", Build.VERSION.SDK_INT)
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("hardware", Build.HARDWARE)
            put("board", Build.BOARD)
            put("app_version", "1.0.0")
            put("timestamp", System.currentTimeMillis())
        }
    }

    /**
     * Create sensor capabilities array
     */
    private fun createCapabilitiesArray(): JSONArray {
        return JSONArray().apply {
            put(
                JSONObject().apply {
                    put("sensor", "rgb_camera")
                    put("type", "video")
                    put("resolution", "1920x1080")
                    put("framerate", 30)
                    put("format", "mp4")
                    put(
                        "features",
                        JSONArray().apply {
                            put("high_resolution")
                            put("samsung_raw_dng")
                            put("4k_recording")
                        },
                    )
                },
            )

            put(
                JSONObject().apply {
                    put("sensor", "thermal_camera")
                    put("type", "thermal")
                    put("hardware", "topdon_tc001")
                    put("accuracy", "±2°C")
                    put("format", "csv")
                    put(
                        "features",
                        JSONArray().apply {
                            put("hardware_calibrated")
                            put("color_palettes")
                            put("temperature_compensation")
                        },
                    )
                },
            )

            put(
                JSONObject().apply {
                    put("sensor", "gsr")
                    put("type", "physiological")
                    put("hardware", "shimmer3_gsr_plus")
                    put("precision", "12_bit_adc")
                    put("sampling_rate", 128)
                    put("format", "csv")
                    put(
                        "features",
                        JSONArray().apply {
                            put("dual_sensor")
                            put("ble_communication")
                            put("scientific_accuracy")
                        },
                    )
                },
            )

            put(
                JSONObject().apply {
                    put("sensor", "audio")
                    put("type", "audio")
                    put("sampling_rate", 44100)
                    put("channels", 2)
                    put("format", "m4a")
                    put(
                        "features",
                        JSONArray().apply {
                            put("stereo_recording")
                            put("aac_encoding")
                            put("high_quality")
                        },
                    )
                },
            )
        }
    }

    /**
     * Create protocol features array
     */
    private fun createFeaturesArray(): JSONArray {
        return JSONArray().apply {
            put("time_synchronization")
            put("flash_synchronization")
            put("session_management")
            put("heartbeat_monitoring")
            put("automatic_reconnection")
            put("status_streaming")
            put("error_reporting")
            put("multi_modal_coordination")
        }
    }

    /**
     * Create session start message
     */
    fun createSessionStartMessage(sessionId: String, timestamp: Long): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "command")
            put("command", MessageTypes.SESSION_START)
            put("session_id", sessionId)
            put("timestamp", timestamp)
            put("message_id", generateMessageId())
        }
    }

    /**
     * Create session stop message
     */
    fun createSessionStopMessage(sessionId: String): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "command")
            put("command", MessageTypes.SESSION_STOP)
            put("session_id", sessionId)
            put("timestamp", System.currentTimeMillis())
            put("message_id", generateMessageId())
        }
    }

    /**
     * Create flash synchronization message
     */
    fun createFlashSyncMessage(flashId: String, timestampNs: Long): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "command")
            put("command", MessageTypes.SYNC_FLASH)
            put("flash_id", flashId)
            put("timestamp_ns", timestampNs)
            put("timestamp_ms", timestampNs / 1_000_000L)
            put("message_id", generateMessageId())
        }
    }

    /**
     * Create comprehensive status update
     */
    fun createStatusUpdate(context: Context, sessionId: String? = null): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "status")
            put("command", MessageTypes.STATUS_UPDATE)
            put("device_id", "${Build.MODEL}_${System.currentTimeMillis()}")
            put("timestamp", System.currentTimeMillis())
            put("message_id", generateMessageId())

            put("battery_level", getBatteryLevel(context))
            put("storage_free_mb", getAvailableStorageMB(context))
            put("memory_free_mb", getAvailableMemoryMB())
            put("cpu_usage", getCpuUsage())

            if (sessionId != null) {
                put("session_id", sessionId)
                put("recording_status", "active")
            } else {
                put("recording_status", "idle")
            }

            put("sensors", createSensorStatusArray())
        }
    }

    /**
     * Create heartbeat message
     */
    fun createHeartbeatMessage(): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "heartbeat")
            put("timestamp", System.currentTimeMillis())
            put("message_id", generateMessageId())
            put("device_id", "${Build.MODEL}_heartbeat")
        }
    }

    /**
     * Create error report message
     */
    fun createErrorReport(error: String, details: String? = null, sessionId: String? = null): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "error")
            put("command", MessageTypes.ERROR_REPORT)
            put("error", error)
            put("timestamp", System.currentTimeMillis())
            put("message_id", generateMessageId())

            if (details != null) {
                put("details", details)
            }

            if (sessionId != null) {
                put("session_id", sessionId)
            }

            put(
                "device_info",
                JSONObject().apply {
                    put("model", Build.MODEL)
                    put("android_version", Build.VERSION.RELEASE)
                },
            )
        }
    }

    /**
     * Create sensor status array
     */
    private fun createSensorStatusArray(): JSONArray {
        return JSONArray().apply {
            put(
                JSONObject().apply {
                    put("sensor", "rgb_camera")
                    put("status", "available")
                    put("connection", "ready")
                },
            )
            put(
                JSONObject().apply {
                    put("sensor", "thermal_camera")
                    put("status", "available")
                    put("connection", "usb_ready")
                },
            )
            put(
                JSONObject().apply {
                    put("sensor", "gsr")
                    put("status", "available")
                    put("connection", "ble_ready")
                },
            )
            put(
                JSONObject().apply {
                    put("sensor", "audio")
                    put("status", "available")
                    put("connection", "ready")
                },
            )
        }
    }

    /**
     * Generate unique message ID
     */
    private fun generateMessageId(): String {
        return "msg_${System.nanoTime()}"
    }

    /**
     * Get battery level (simplified - would need actual implementation)
     */
    private fun getBatteryLevel(context: Context): Int {
        return 85
    }

    /**
     * Get available storage in MB
     */
    private fun getAvailableStorageMB(context: Context): Long {
        return try {
            val files = context.filesDir
            files.freeSpace / (1024 * 1024)
        } catch (e: Exception) {
            -1L
        }
    }

    /**
     * Get available memory in MB
     */
    private fun getAvailableMemoryMB(): Long {
        return try {
            val runtime = Runtime.getRuntime()
            (runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory()) / (1024 * 1024)
        } catch (e: Exception) {
            -1L
        }
    }

    /**
     * Get CPU usage (simplified)
     */
    private fun getCpuUsage(): Double {
        return 25.0
    }

    /**
     * Parse and validate enhanced protocol message
     */
    fun parseMessage(messageStr: String): JSONObject? {
        return try {
            val message = JSONObject(messageStr)

            if (!message.has("type")) {
                Log.w(TAG, "Message missing 'type' field")
                return null
            }

            if (message.has("v")) {
                val version = message.getString("v")
                if (version != "1.0" && version != PROTOCOL_VERSION) {
                    Log.w(TAG, "Unsupported protocol version: $version")
                }
            }

            message
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse message: ${e.message}", e)
            null
        }
    }

    /**
     * Create acknowledgment for enhanced protocol message
     */
    fun createAcknowledgment(originalMessage: JSONObject, status: String = "ok", data: JSONObject? = null): JSONObject {
        return JSONObject().apply {
            put("v", PROTOCOL_VERSION)
            put("type", "ack")
            put("status", status)
            put("timestamp", System.currentTimeMillis())

            if (originalMessage.has("message_id")) {
                put("ack_message_id", originalMessage.getString("message_id"))
            }

            if (data != null) {
                put("data", data)
            }
        }
    }
}
