package com.topdon.tc001

import android.app.Application
import android.util.Log

class App : Application() {

    companion object {
        lateinit var instance: App
            private set
        private const val TAG = "App"
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        
        Log.d(TAG, "Application initialized with enhanced GSR sensor support")
    }

    fun initWebSocket() {
        // Initialize WebSocket connection for TC001 devices
    }
}