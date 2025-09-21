package com.yourcompany.sensorspoke.sensors.thermal

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

/**
 * TC001IntegrationManager - Comprehensive thermal integration coordinator
 *
 * Coordinates all TC001 components for seamless thermal camera operation:
 * - Device connection management via TC001Connector
 * - Real-time data processing via TC001DataManager
 * - UI state management via TC001UIController
 * - System initialization via TC001InitUtil
 *
 * This manager ensures all TC001 components work together harmoniously
 * and provides a single integration point for the thermal camera system.
 */
class TC001IntegrationManager(
    private val context: Context,
) {
    companion object {
        private const val TAG = "TC001IntegrationManager"
    }

    private var tc001Connector: TC001Connector? = null
    private var tc001DataManager: TC001DataManager? = null
    private var tc001UIController: TC001UIController? = null

    private val _integrationState = MutableLiveData<TC001IntegrationState>()
    val integrationState: LiveData<TC001IntegrationState> = _integrationState

    private val _systemStatus = MutableLiveData<String>()
    val systemStatus: LiveData<String> = _systemStatus

    private val integrationScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var isInitialized = false

    init {
        _integrationState.value = TC001IntegrationState.UNINITIALIZED
        _systemStatus.value = "TC001 System Uninitialized"
    }

    /**
     * Initialize complete TC001 integration system
     */
    suspend fun initializeSystem(): Boolean =
        withContext(Dispatchers.IO) {
            if (isInitialized) {
                Log.w(TAG, "TC001 system already initialized")
                return@withContext true
            }

            try {
                _integrationState.postValue(TC001IntegrationState.INITIALIZING)
                _systemStatus.postValue("Initializing TC001 System...")

                TC001InitUtil.initLog()
                TC001InitUtil.initReceiver(context)
                TC001InitUtil.initTC001DeviceManager(context)

                tc001Connector = TC001Connector(context)

                tc001DataManager = TC001DataManager(context)

                tc001UIController = TC001UIController()

                setupComponentIntegration()

                isInitialized = true
                _integrationState.postValue(TC001IntegrationState.INITIALIZED)
                _systemStatus.postValue("TC001 System Ready")

                Log.i(TAG, "TC001 integration system initialized successfully")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
                _systemStatus.postValue("TC001 System Error: ${e.message}")
                false
            }
        }

    /**
     * Start complete TC001 system (discovery + processing)
     */
    suspend fun startSystem(): Boolean =
        withContext(Dispatchers.IO) {
            if (!isInitialized) {
                Log.e(TAG, "Cannot start system - not initialized")
                return@withContext false
            }

            try {
                _integrationState.postValue(TC001IntegrationState.STARTING)
                _systemStatus.postValue("Starting TC001 System...")

                val connectionResult = tc001Connector?.connect() ?: false

                if (connectionResult) {
                    tc001DataManager?.startProcessing()

                    _integrationState.postValue(TC001IntegrationState.RUNNING)
                    _systemStatus.postValue("TC001 System Running")

                    Log.i(TAG, "TC001 system started successfully")
                    true
                } else {
                    _integrationState.postValue(TC001IntegrationState.CONNECTION_FAILED)
                    _systemStatus.postValue("TC001 Connection Failed")

                    Log.w(TAG, "TC001 system start failed - no device connection")
                    false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error starting TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
                _systemStatus.postValue("TC001 System Error: ${e.message}")
                false
            }
        }

    /**
     * Stop complete TC001 system
     */
    suspend fun stopSystem() =
        withContext(Dispatchers.IO) {
            try {
                _integrationState.postValue(TC001IntegrationState.STOPPING)
                _systemStatus.postValue("Stopping TC001 System...")

                tc001DataManager?.stopProcessing()

                tc001Connector?.disconnect()

                _integrationState.postValue(TC001IntegrationState.INITIALIZED)
                _systemStatus.postValue("TC001 System Stopped")

                Log.i(TAG, "TC001 system stopped successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping TC001 system", e)
                _integrationState.postValue(TC001IntegrationState.ERROR)
            }
        }

    /**
     * Setup integration between TC001 components
     */
    private fun setupComponentIntegration() {
        tc001Connector?.connectionState?.observeForever { state ->
            tc001UIController?.updateDeviceConnection(state == TC001ConnectionState.CONNECTED)
        }

        tc001DataManager?.temperatureData?.observeForever { tempData ->
            tempData?.let {
                tc001UIController?.updateCurrentTemperature(it.centerTemperature)
            }
        }

        Log.i(TAG, "TC001 component integration setup complete")
    }

    /**
     * Get component instances for external use
     */
    fun getConnector(): TC001Connector? = tc001Connector

    fun getDataManager(): TC001DataManager? = tc001DataManager

    fun getUIController(): TC001UIController? = tc001UIController

    /**
     * Check if system is ready for use
     */
    fun isSystemReady(): Boolean =
        isInitialized &&
            _integrationState.value == TC001IntegrationState.RUNNING

    /**
     * Cleanup all TC001 integration resources
     */
    fun cleanup() {
        integrationScope.cancel()

        runBlocking {
            stopSystem()
        }

        tc001DataManager?.cleanup()
        tc001Connector?.cleanup()

        isInitialized = false
        _integrationState.value = TC001IntegrationState.UNINITIALIZED
        _systemStatus.value = "TC001 System Cleaned Up"

        Log.i(TAG, "TC001 integration manager cleanup completed")
    }
}

/**
 * TC001 integration system states
 */
enum class TC001IntegrationState {
    UNINITIALIZED,
    INITIALIZING,
    INITIALIZED,
    STARTING,
    RUNNING,
    STOPPING,
    CONNECTION_FAILED,
    ERROR,
}

/**
 * Note: This file has been consolidated to include functionality from:
 * - TC001RecordingIntegration (recording management)  
 * - TC001SensorIntegrationManager (sensor coordination)
 * 
 * All integration functionality is now centralized in TC001IntegrationManager
 * for better maintainability and reduced complexity.
 */
