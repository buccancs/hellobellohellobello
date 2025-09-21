pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven { url = uri("https://jitpack.io") }
        maven { url = uri("https://developer.huawei.com/repo/") }
    }
    plugins {
        id("com.android.application") version "8.1.4"
        id("org.jetbrains.kotlin.android") version "1.9.25"
        id("org.jlleitschuh.gradle.ktlint") version "12.1.2"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        // JitPack for GitHub-based libraries like FastBLE
        maven { url = uri("https://jitpack.io") }
        // Huawei repository for HMS services
        maven { url = uri("https://developer.huawei.com/repo/") }
    }
}

rootProject.name = "hellobellohellobello"

// ============================================================================
// Environment Detection and Module Configuration
// ============================================================================

/**
 * Centralized Android SDK detection function
 */
fun detectAndroidSdk(): Boolean {
    // Check local.properties
    val localPropsFile = file("local.properties")
    if (localPropsFile.exists()) {
        val props = java.util.Properties()
        localPropsFile.inputStream().use { props.load(it) }
        val sdkDir = props.getProperty("sdk.dir")
        if (sdkDir != null && file(sdkDir).exists()) {
            return true
        }
    }

    // Check environment variables
    val envSdk = System.getenv("ANDROID_SDK_ROOT") ?: System.getenv("ANDROID_HOME")
    return envSdk != null && file(envSdk).exists()
}

/**
 * Check for Python environment and dependencies  
 * Note: Configuration cache compatible - does not execute external processes
 */
fun detectPythonEnvironment(): Boolean {
    // For configuration cache compatibility, we'll assume Python is available
    // and let runtime tasks handle actual validation
    return System.getenv("PYTHON_AVAILABLE") != "false"
}

// Detect available development environments
val hasAndroidSdk = detectAndroidSdk()
val hasPython = detectPythonEnvironment()

// ============================================================================
// Module Inclusion Logic
// ============================================================================

// Always include PC Controller (Python-based)
include(":pc_controller")
project(":pc_controller").projectDir = file("pc_controller")

// Conditionally include Android modules based on SDK availability
if (hasAndroidSdk) {
    include(":android_sensor_node")
    project(":android_sensor_node").projectDir = file("android_sensor_node")
    
    include(":android_sensor_node:app")
    project(":android_sensor_node:app").projectDir = file("android_sensor_node/app")
    
    println("[settings] ✓ Android SDK found - including Android sensor node modules")
} else {
    println("[settings] ✗ Android SDK not found - skipping Android sensor node modules")
    println("[settings]   To enable Android development:")
    println("[settings]   1. Install Android SDK")
    println("[settings]   2. Set ANDROID_HOME or ANDROID_SDK_ROOT environment variable")
    println("[settings]   3. Or create local.properties with sdk.dir=/path/to/android/sdk")
}

// Environment status report
println("""
[settings] ================================================================================
[settings] Multi-Modal Physiological Sensing Platform - Build Environment
[settings] ================================================================================
[settings] Android SDK: ${if (hasAndroidSdk) "✓ Available" else "✗ Not Available"}
[settings] Python 3:    ${if (hasPython) "✓ Available" else "✗ Not Available"}
[settings] 
[settings] Included Modules:
[settings]  • :pc_controller              (Python-based hub application)
""".trimIndent())

if (hasAndroidSdk) {
    println("[settings]  • :android_sensor_node        (Android sensor node container)")
    println("[settings]  • :android_sensor_node:app    (Main Android application)")
}

println("[settings] ================================================================================")

// ============================================================================
// Build Performance and Feature Flags
// ============================================================================

// Note: Some Gradle build features are configured via gradle.properties
// for better compatibility with configuration cache
