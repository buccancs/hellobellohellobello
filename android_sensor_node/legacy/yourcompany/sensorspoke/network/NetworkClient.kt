package com.yourcompany.sensorspoke.network

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import java.io.IOException
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * NetworkClient encapsulates Android NSD (Zeroconf) service registration
 * and TCP communication for the Sensor Spoke. Supports both service discovery
 * and reliable message transmission to PC Hub.
 */
class NetworkClient(
    private val context: Context,
) {
    companion object {
        private const val TAG = "NetworkClient"
        private const val DEFAULT_TIMEOUT_MS = 10000
        private const val CONNECTION_RETRY_DELAY_MS = 1000L
    }

    private var nsdManager: NsdManager = context.getSystemService(Context.NSD_SERVICE) as NsdManager
    private var registrationListener: NsdManager.RegistrationListener? = null

    // Connection state
    private val socket = AtomicReference<Socket?>()
    private val isConnected = AtomicBoolean(false)
    private val serverAddress = AtomicReference<InetSocketAddress?>()

    // Connection configuration
    var connectionTimeoutMs: Int = DEFAULT_TIMEOUT_MS
    var autoReconnect: Boolean = true

    /**
     * Register this device as an NSD service for PC Hub discovery.
     */
    fun register(
        type: String,
        name: String,
        port: Int,
    ) {
        val sanitizedType = if (type.endsWith(".local.")) type else "$type.local."
        val info = NsdServiceInfo()
        // Use explicit Java-style setters to avoid Kotlin property mutability issues
        info.serviceType = sanitizedType
        info.serviceName = name
        info.port = port

        val listener =
            object : NsdManager.RegistrationListener {
                override fun onServiceRegistered(nsdServiceInfo: NsdServiceInfo) {
                    Log.i(TAG, "Service registered: ${nsdServiceInfo.serviceName}")
                }

                override fun onRegistrationFailed(
                    serviceInfo: NsdServiceInfo,
                    errorCode: Int,
                ) {
                    Log.e(TAG, "Service registration failed: $errorCode")
                }

                override fun onServiceUnregistered(serviceInfo: NsdServiceInfo) {
                    Log.i(TAG, "Service unregistered: ${serviceInfo.serviceName}")
                }

                override fun onUnregistrationFailed(
                    serviceInfo: NsdServiceInfo,
                    errorCode: Int,
                ) {
                    Log.e(TAG, "Service unregistration failed: $errorCode")
                }
            }
        registrationListener = listener
        nsdManager.registerService(info, NsdManager.PROTOCOL_DNS_SD, listener)
    }

    /**
     * Unregister the NSD service.
     */
    fun unregister() {
        registrationListener?.let {
            try {
                nsdManager.unregisterService(it)
            } catch (e: Exception) {
                Log.w(TAG, "Error unregistering service", e)
            }
        }
        registrationListener = null
    }

    /**
     * Connect to the PC Hub server.
     */
    fun connect(
        host: String,
        port: Int,
    ): Boolean =
        try {
            disconnect() // Close any existing connection

            val newSocket = Socket()
            val address = InetSocketAddress(host, port)

            Log.i(TAG, "Connecting to $host:$port...")
            newSocket.connect(address, connectionTimeoutMs)

            socket.set(newSocket)
            serverAddress.set(address)
            isConnected.set(true)

            Log.i(TAG, "Successfully connected to $host:$port")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to $host:$port", e)
            isConnected.set(false)
            false
        }

    /**
     * Disconnect from the PC Hub server.
     */
    fun disconnect() {
        isConnected.set(false)
        socket.getAndSet(null)?.let { s ->
            try {
                if (!s.isClosed) {
                    s.close()
                }
                Log.i(TAG, "Disconnected from server")
            } catch (e: Exception) {
                Log.w(TAG, "Error closing socket", e)
            }
        }
    }

    /**
     * Send a message to the PC Hub server.
     * @param message The message to send (JSON string)
     * @return true if message was sent successfully, false otherwise
     */
    fun sendMessage(message: String): Boolean {
        val currentSocket = socket.get()
        if (currentSocket == null || currentSocket.isClosed || !isConnected.get()) {
            Log.w(TAG, "Cannot send message - not connected")
            return false
        }

        return try {
            val outputStream: OutputStream = currentSocket.getOutputStream()
            val messageBytes = message.toByteArray(StandardCharsets.UTF_8)

            // Send message with newline delimiter (line-based protocol)
            outputStream.write(messageBytes)
            outputStream.write('\n'.code)
            outputStream.flush()

            Log.d(TAG, "Message sent successfully: ${message.take(100)}...")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Failed to send message", e)
            markConnectionLost()
            false
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error sending message", e)
            false
        }
    }

    /**
     * Attempt to reconnect to the previously connected server.
     * @return true if reconnection succeeded, false otherwise
     */
    fun reconnect(): Boolean {
        val address = serverAddress.get()
        if (address == null) {
            Log.w(TAG, "Cannot reconnect - no previous server address")
            return false
        }

        return connect(address.hostString, address.port)
    }

    /**
     * Check if currently connected to a server.
     */
    fun isConnected(): Boolean {
        val currentSocket = socket.get()
        return isConnected.get() &&
            currentSocket != null &&
            !currentSocket.isClosed &&
            currentSocket.isConnected
    }

    /**
     * Get current server address (if connected).
     */
    fun getServerAddress(): String? = serverAddress.get()?.let { "${it.hostString}:${it.port}" }

    /**
     * Mark connection as lost and trigger cleanup.
     */
    private fun markConnectionLost() {
        if (isConnected.getAndSet(false)) {
            Log.w(TAG, "Connection marked as lost")
            // Don't close socket immediately - let reconnect logic handle it
        }
    }

    /**
     * Get connection status summary for debugging and user display.
     */
    fun getConnectionStatus(): Map<String, Any> =
        mapOf(
            "connected" to isConnected(),
            "server_address" to (getServerAddress() ?: "none"),
            "socket_closed" to (socket.get()?.isClosed ?: true),
            "auto_reconnect" to autoReconnect,
            "timeout_ms" to connectionTimeoutMs,
        )

    /**
     * Get user-friendly connection status for display in UI.
     */
    fun getUserFriendlyStatus(): String =
        when {
            isConnected() -> "Connected to PC Hub: ${getServerAddress()}"
            serverAddress.get() != null -> "Disconnected (attempting to reconnect...)"
            else -> "Not connected to PC Hub"
        }

    // PC Discovery functionality
    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private var resolveListener: NsdManager.ResolveListener? = null

    /**
     * Discover PC Hub services on the network.
     * @param serviceType The service type to discover (e.g., "_gsr-controller._tcp.")
     * @param onDiscovered Callback when a PC Hub is discovered with address and port
     * @param onLost Callback when a PC Hub service is lost
     */
    fun discoverPCHubs(
        serviceType: String = "_gsr-controller._tcp.",
        onDiscovered: (String, String, Int) -> Unit,
        onLost: (String) -> Unit,
    ) {
        stopDiscovery() // Stop any existing discovery

        val sanitizedType = if (serviceType.endsWith(".local.")) serviceType else "$serviceType.local."

        discoveryListener =
            object : NsdManager.DiscoveryListener {
                override fun onStartDiscoveryFailed(
                    serviceType: String,
                    errorCode: Int,
                ) {
                    Log.e(TAG, "PC Hub discovery start failed: $errorCode")
                }

                override fun onStopDiscoveryFailed(
                    serviceType: String,
                    errorCode: Int,
                ) {
                    Log.e(TAG, "PC Hub discovery stop failed: $errorCode")
                }

                override fun onDiscoveryStarted(serviceType: String) {
                    Log.i(TAG, "PC Hub discovery started for: $serviceType")
                }

                override fun onDiscoveryStopped(serviceType: String) {
                    Log.i(TAG, "PC Hub discovery stopped for: $serviceType")
                }

                override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                    Log.d(TAG, "PC Hub service found: ${serviceInfo.serviceName}")

                    // Resolve the service to get IP and port
                    resolveListener =
                        object : NsdManager.ResolveListener {
                            override fun onResolveFailed(
                                serviceInfo: NsdServiceInfo,
                                errorCode: Int,
                            ) {
                                Log.w(TAG, "PC Hub resolve failed: $errorCode")
                            }

                            override fun onServiceResolved(serviceInfo: NsdServiceInfo) {
                                Log.i(TAG, "PC Hub resolved: ${serviceInfo.serviceName} @ ${serviceInfo.host}:${serviceInfo.port}")
                                onDiscovered(
                                    serviceInfo.serviceName,
                                    serviceInfo.host.hostAddress ?: "unknown",
                                    serviceInfo.port,
                                )
                            }
                        }

                    nsdManager.resolveService(serviceInfo, resolveListener)
                }

                override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                    Log.d(TAG, "PC Hub service lost: ${serviceInfo.serviceName}")
                    onLost(serviceInfo.serviceName)
                }
            }

        nsdManager.discoverServices(sanitizedType, NsdManager.PROTOCOL_DNS_SD, discoveryListener)
    }

    /**
     * Stop PC Hub discovery.
     */
    fun stopDiscovery() {
        discoveryListener?.let { listener ->
            try {
                nsdManager.stopServiceDiscovery(listener)
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping PC Hub discovery", e)
            }
        }
        discoveryListener = null
        resolveListener = null
    }

    /**
     * Enhanced connection method that automatically discovers and connects to first available PC Hub.
     * @param serviceType The service type to discover
     * @param onConnected Callback when connection is established
     * @param onFailed Callback when connection fails
     */
    fun autoConnectToPCHub(
        serviceType: String = "_gsr-controller._tcp.",
        onConnected: (String, String, Int) -> Unit,
        onFailed: (String) -> Unit,
    ) {
        Log.i(TAG, "Starting automatic PC Hub connection...")

        discoverPCHubs(
            serviceType = serviceType,
            onDiscovered = { name, host, port ->
                Log.i(TAG, "Attempting auto-connection to $name at $host:$port")

                // Stop discovery once we find a service
                stopDiscovery()

                // Attempt connection
                if (connect(host, port)) {
                    Log.i(TAG, "Auto-connection successful to $name")
                    onConnected(name, host, port)
                } else {
                    Log.w(TAG, "Auto-connection failed to $name")
                    onFailed("Failed to connect to discovered PC Hub: $name")
                }
            },
            onLost = { name ->
                Log.d(TAG, "PC Hub service lost during discovery: $name")
            },
        )

        // Set a timeout for discovery
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            if (discoveryListener != null && !isConnected()) {
                stopDiscovery()
                onFailed("No PC Hub found on network within timeout")
            }
        }, 10000) // 10 second timeout
    }
}
