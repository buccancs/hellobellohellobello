package com.topdon.tc001

import android.app.Application

class App : Application() {

    companion object {
        lateinit var instance: App
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    fun initWebSocket() {
        // Initialize WebSocket connection for TC001 devices
    }
}