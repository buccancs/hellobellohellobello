package com.yourcompany.sensorspoke.ui.models

import java.io.File
import java.util.Date

/**
 * Data class representing information about a recording session
 */
data class SessionInfo(
    val name: String,
    val dateTime: Date,
    val sizeBytes: Long,
    val details: String,
    val directory: File,
)
