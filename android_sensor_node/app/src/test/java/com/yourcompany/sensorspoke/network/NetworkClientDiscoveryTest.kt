/**
 * Tests for NetworkClient PC discovery enhancements.
 */
package com.yourcompany.sensorspoke.network

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.runs
import io.mockk.slot
import io.mockk.spyk
import io.mockk.verify
import org.junit.Before
import org.junit.Test
import java.net.InetAddress

class NetworkClientDiscoveryTest {
    private lateinit var mockContext: Context
    private lateinit var mockNsdManager: NsdManager
    private lateinit var networkClient: NetworkClient

    @Before
    fun setup() {
        mockContext = mockk()
        mockNsdManager = mockk()

        every { mockContext.getSystemService(Context.NSD_SERVICE) } returns mockNsdManager
        every { mockNsdManager.registerService(any(), any(), any()) } just runs
        every { mockNsdManager.unregisterService(any()) } just runs
        every { mockNsdManager.discoverServices(any(), any(), any()) } just runs
        every { mockNsdManager.stopServiceDiscovery(any()) } just runs
        every { mockNsdManager.resolveService(any(), any()) } just runs

        networkClient = NetworkClient(mockContext)
    }

    @Test
    fun testDiscoverPCHubs_StartsDiscovery() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        networkClient.discoverPCHubs(
            serviceType = "_gsr-controller._tcp.",
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        verify { mockNsdManager.discoverServices("_gsr-controller._tcp.local.", NsdManager.PROTOCOL_DNS_SD, any()) }
    }

    @Test
    fun testDiscoverPCHubs_HandlesServiceFound() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        val serviceInfo =
            mockk<NsdServiceInfo> {
                every { serviceName } returns "TestPC"
            }

        val slot = slot<NsdManager.DiscoveryListener>()
        every { mockNsdManager.discoverServices(any(), any(), capture(slot)) } just runs

        networkClient.discoverPCHubs(
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        slot.captured.onServiceFound(serviceInfo)

        verify { mockNsdManager.resolveService(serviceInfo, any()) }
    }

    @Test
    fun testDiscoverPCHubs_HandlesServiceResolved() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        val serviceInfo =
            mockk<NsdServiceInfo> {
                every { serviceName } returns "TestPC"
                every { host } returns InetAddress.getByName("192.168.1.100")
                every { port } returns 8080
            }

        val discoverySlot = slot<NsdManager.DiscoveryListener>()
        val resolveSlot = slot<NsdManager.ResolveListener>()

        every { mockNsdManager.discoverServices(any(), any(), capture(discoverySlot)) } just runs
        every { mockNsdManager.resolveService(any(), capture(resolveSlot)) } just runs

        networkClient.discoverPCHubs(
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        discoverySlot.captured.onServiceFound(serviceInfo)
        resolveSlot.captured.onServiceResolved(serviceInfo)

        verify { onDiscovered("TestPC", "192.168.1.100", 8080) }
    }

    @Test
    fun testDiscoverPCHubs_HandlesServiceLost() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        val serviceInfo =
            mockk<NsdServiceInfo> {
                every { serviceName } returns "TestPC"
            }

        val slot = slot<NsdManager.DiscoveryListener>()
        every { mockNsdManager.discoverServices(any(), any(), capture(slot)) } just runs

        networkClient.discoverPCHubs(
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        slot.captured.onServiceLost(serviceInfo)

        verify { onLost("TestPC") }
    }

    @Test
    fun testStopDiscovery_StopsActiveDiscovery() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        val slot = slot<NsdManager.DiscoveryListener>()
        every { mockNsdManager.discoverServices(any(), any(), capture(slot)) } just runs

        networkClient.discoverPCHubs(
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        networkClient.stopDiscovery()

        verify { mockNsdManager.stopServiceDiscovery(slot.captured) }
    }

    @Test
    fun testAutoConnectToPCHub_AttemptsConnection() {
        val onConnected = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onFailed = mockk<(String) -> Unit>(relaxed = true)

        val serviceInfo =
            mockk<NsdServiceInfo> {
                every { serviceName } returns "TestPC"
                every { host } returns InetAddress.getByName("192.168.1.100")
                every { port } returns 8080
            }

        val discoverySlot = slot<NsdManager.DiscoveryListener>()
        val resolveSlot = slot<NsdManager.ResolveListener>()

        every { mockNsdManager.discoverServices(any(), any(), capture(discoverySlot)) } just runs
        every { mockNsdManager.resolveService(any(), capture(resolveSlot)) } just runs

        val networkClientSpy = spyk(networkClient)
        every { networkClientSpy.connect("192.168.1.100", 8080) } returns true

        networkClientSpy.autoConnectToPCHub(
            onConnected = onConnected,
            onFailed = onFailed,
        )

        discoverySlot.captured.onServiceFound(serviceInfo)
        resolveSlot.captured.onServiceResolved(serviceInfo)

        verify { networkClientSpy.connect("192.168.1.100", 8080) }
        verify { onConnected("TestPC", "192.168.1.100", 8080) }
        verify { mockNsdManager.stopServiceDiscovery(any()) }
    }

    @Test
    fun testAutoConnectToPCHub_HandlesConnectionFailure() {
        val onConnected = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onFailed = mockk<(String) -> Unit>(relaxed = true)

        val serviceInfo =
            mockk<NsdServiceInfo> {
                every { serviceName } returns "TestPC"
                every { host } returns InetAddress.getByName("192.168.1.100")
                every { port } returns 8080
            }

        val discoverySlot = slot<NsdManager.DiscoveryListener>()
        val resolveSlot = slot<NsdManager.ResolveListener>()

        every { mockNsdManager.discoverServices(any(), any(), capture(discoverySlot)) } just runs
        every { mockNsdManager.resolveService(any(), capture(resolveSlot)) } just runs

        val networkClientSpy = spyk(networkClient)
        every { networkClientSpy.connect("192.168.1.100", 8080) } returns false

        networkClientSpy.autoConnectToPCHub(
            onConnected = onConnected,
            onFailed = onFailed,
        )

        discoverySlot.captured.onServiceFound(serviceInfo)
        resolveSlot.captured.onServiceResolved(serviceInfo)

        verify { networkClientSpy.connect("192.168.1.100", 8080) }
        verify { onFailed(match { it.contains("Failed to connect to discovered PC Hub: TestPC") }) }
        verify(exactly = 0) { onConnected(any(), any(), any()) }
    }

    @Test
    fun testDiscoverPCHubs_SanitizesServiceType() {
        val onDiscovered = mockk<(String, String, Int) -> Unit>(relaxed = true)
        val onLost = mockk<(String) -> Unit>(relaxed = true)

        networkClient.discoverPCHubs(
            serviceType = "_gsr-controller._tcp",
            onDiscovered = onDiscovered,
            onLost = onLost,
        )

        verify { mockNsdManager.discoverServices("_gsr-controller._tcp.local.", NsdManager.PROTOCOL_DNS_SD, any()) }
    }
}
