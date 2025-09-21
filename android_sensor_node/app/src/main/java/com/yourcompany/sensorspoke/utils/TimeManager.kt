package com.yourcompany.sensorspoke.utils

import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

/**
 * TimeManager provides access to a monotonic time source to timestamp data.
 * Adds lightweight time synchronization with PC server via UDP echo of
 * time.monotonic_ns (server-side) to fulfill FR3/NFR2.
 */
object TimeManager {
    @Volatile
    private var offsetNs: Long = 0L

    /**
     * Returns a monotonic timestamp in nanoseconds.
     */
    @JvmStatic
    fun nowNanos(): Long = System.nanoTime()

    /**
     * Returns current time adjusted by last known offset computed via sync_with_server.
     */
    @JvmStatic
    fun getSyncedTimestamp(): Long {
        val now = System.nanoTime()
        return now + offsetNs
    }

    /**
     * Performs a single-shot sync with the server on a background thread.
     * The server must reply with ASCII-encoded nanoseconds (time.monotonic_ns).
     *
     * offset = (T_server + (T2 - T1) / 2) - T2
     */
    @JvmStatic
    fun sync_with_server(
        serverIp: String,
        serverPort: Int,
        timeoutMillis: Int = 1500,
        onComplete: ((Boolean) -> Unit)? = null,
    ) {
        Thread {
            var socket: DatagramSocket? = null
            try {
                socket = DatagramSocket()
                socket.soTimeout = timeoutMillis
                val addr = InetAddress.getByName(serverIp)

                val sendBuf = byteArrayOf(1)
                val sendPkt = DatagramPacket(sendBuf, sendBuf.size, addr, serverPort)

                val t1 = System.nanoTime()
                socket.send(sendPkt)

                val recvBuf = ByteArray(64)
                val recvPkt = DatagramPacket(recvBuf, recvBuf.size)
                socket.receive(recvPkt)
                val t2 = System.nanoTime()

                val serverStr = String(recvPkt.data, 0, recvPkt.length, Charsets.US_ASCII).trim()
                val tServer = serverStr.toLong()

                val offset = (tServer + ((t2 - t1) / 2)) - t2
                offsetNs = offset
                onComplete?.invoke(true)
            } catch (_: Exception) {
                onComplete?.invoke(false)
            } finally {
                try {
                    socket?.close()
                } catch (_: Exception) {
                }
            }
        }.start()
    }
}
