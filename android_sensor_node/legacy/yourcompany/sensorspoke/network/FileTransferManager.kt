package com.yourcompany.sensorspoke.network

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * FileTransferManager (Phase 5):
 * - Compress a session directory into a ZIP stream
 * - Transfer it to the PC Hub's FileReceiver over TCP
 */
class FileTransferManager(
    private val context: Context? = null,
    private val sessionsRootOverride: File? = null,
    private val flashEventsFileOverride: File? = null,
    private val deviceIdOverride: String? = null,
) {
    private fun sessionRoot(): File {
        sessionsRootOverride?.let { return it }
        val ctx = context ?: throw IllegalStateException("sessionsRootOverride is required when context is null")
        val base: File? = ctx.getExternalFilesDir(null)
        return File(base ?: ctx.filesDir, "sessions")
    }

    private fun sessionDir(sessionId: String): File = File(sessionRoot(), sessionId)

    /**
     * Transfer the given sessionId directory to the host:port receiver.
     * Streams a ZIP over the socket without creating a temp file.
     */
    suspend fun transferSession(
        sessionId: String,
        host: String,
        port: Int,
    ) = withContext(Dispatchers.IO) {
        val dir = sessionDir(sessionId)
        if (!dir.exists() || !dir.isDirectory) throw IllegalArgumentException("Session not found: ${dir.absolutePath}")

        val deviceId = deviceIdOverride ?: (Build.MODEL ?: "device").replace(" ", "_")
        val socket = Socket()
        // 2s connect timeout; 5s read timeout to avoid hangs in tests/CI
        socket.connect(InetSocketAddress(host, port), 2000)
        socket.tcpNoDelay = true
        socket.soTimeout = 5000
        socket.use {
            val out = BufferedOutputStream(it.getOutputStream())
            // Send header JSON line first (avoid org.json for JVM tests)
            val headerLine =
                "{" +
                    "\"session_id\":\"$sessionId\"," +
                    "\"device_id\":\"$deviceId\"," +
                    "\"filename\":\"${deviceId}_data.zip\"" +
                    "}"
            out.write((headerLine + "\n").toByteArray(Charsets.UTF_8))
            out.flush()
            // Now stream ZIP content
            ZipOutputStream(out).use { zos ->
                zipDirectoryContents(dir, dir, zos)
                // Also include flash_sync_events.csv if present (override or derived)
                try {
                    val flash: File? =
                        flashEventsFileOverride ?: run {
                            val ctx = context
                            if (ctx != null) {
                                val root = ctx.getExternalFilesDir(null) ?: ctx.filesDir
                                File(root, "flash_sync_events.csv")
                            } else {
                                null
                            }
                        }
                    if (flash != null && flash.exists() && flash.isFile) {
                        val entry = ZipEntry("flash_sync_events.csv")
                        entry.time = flash.lastModified()
                        zos.putNextEntry(entry)
                        FileInputStream(flash).use { fis ->
                            val buf = ByteArray(64 * 1024)
                            while (true) {
                                val n = fis.read(buf)
                                if (n <= 0) break
                                zos.write(buf, 0, n)
                            }
                        }
                        zos.closeEntry()
                    }
                } catch (_: Exception) {
                    // best-effort inclusion
                }
                zos.finish()
                zos.flush()
            }
            out.flush()
        }
    }

    private fun zipDirectoryContents(
        root: File,
        src: File,
        zos: ZipOutputStream,
    ) {
        val files = src.listFiles() ?: return
        for (f in files) {
            if (f.isDirectory) {
                zipDirectoryContents(root, f, zos)
            } else {
                val relPath = root.toURI().relativize(f.toURI()).path
                val entry = ZipEntry(relPath)
                entry.time = f.lastModified()
                zos.putNextEntry(entry)
                FileInputStream(f).use { fis ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = fis.read(buf)
                        if (n <= 0) break
                        zos.write(buf, 0, n)
                    }
                }
                zos.closeEntry()
            }
        }
    }
}
