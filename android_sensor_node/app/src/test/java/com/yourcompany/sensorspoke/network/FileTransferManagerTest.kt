package com.yourcompany.sensorspoke.network

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.runBlocking
import org.junit.Test
import java.io.BufferedInputStream
import java.io.InputStream
import java.net.ServerSocket
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

class FileTransferManagerTest {
    @Test
    fun transferSession_streamsHeaderAndZipEntries() {
        runBlocking {
            val sessionsRoot = Files.createTempDirectory("sessions_root").toFile()
            val sessionId = "session_abc"
            val sessionDir = java.io.File(sessionsRoot, sessionId).apply { mkdirs() }
            val sub1 = java.io.File(sessionDir, "rgb").apply { mkdirs() }
            val sub2 = java.io.File(sessionDir, "gsr").apply { mkdirs() }
            java.io.File(sub1, "file1.txt").apply { writeText("hello") }
            java.io.File(sub2, "file2.txt").apply { writeText("world") }

            val server = ServerSocket(0)
            val port = server.localPort
            val pool = Executors.newSingleThreadExecutor()
            val latch = CountDownLatch(1)
            val headerHolder = arrayOfNulls<String>(1)
            val entries = mutableListOf<String>()

            pool.submit {
                server.use { ss ->
                    ss.accept().use { socket ->
                        readSession(socket.getInputStream(), headerHolder, entries)
                        latch.countDown()
                    }
                }
            }

            val ftm =
                FileTransferManager(
                    context = null,
                    sessionsRootOverride = sessionsRoot,
                    flashEventsFileOverride = null,
                    deviceIdOverride = "TestDevice",
                )
            ftm.transferSession(sessionId, "127.0.0.1", port)

            latch.await(5, TimeUnit.SECONDS)
            val header = headerHolder[0]
            require(header != null) { "Header was null" }
            assertThat(header.trim().startsWith("{")).isTrue()
            assertThat(header.contains("\"session_id\":\"$sessionId\"")).isTrue()
            assertThat(header.contains("\"device_id\":\"TestDevice\"")).isTrue()
            assertThat(header.contains("\"filename\":\"TestDevice_data.zip\"")).isTrue()

            assertThat(entries.any { it.endsWith("rgb/file1.txt") }).isTrue()
            assertThat(entries.any { it.endsWith("gsr/file2.txt") }).isTrue()

            pool.shutdownNow()
            sessionsRoot.deleteRecursively()
        }
    }

    private fun readSession(
        `in`: InputStream,
        headerOut: Array<String?>,
        entries: MutableList<String>,
    ) {
        val bis = BufferedInputStream(`in`)
        val headerSb = StringBuilder()
        while (true) {
            val ch = bis.read()
            if (ch == -1) throw IllegalStateException("Stream ended before header newline")
            if (ch.toChar() == '\n') break
            headerSb.append(ch.toChar())
        }
        headerOut[0] = headerSb.toString()
        ZipInputStream(bis).use { zis ->
            while (true) {
                val e: ZipEntry = zis.nextEntry ?: break
                entries.add(e.name)
                val buffer = ByteArray(4096)
                while (true) {
                    val n = zis.read(buffer)
                    if (n <= 0) break
                }
                zis.closeEntry()
            }
        }
    }
}
