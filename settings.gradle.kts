pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        id("com.android.application") version "8.5.2"
        id("org.jetbrains.kotlin.android") version "1.9.25"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // JitPack for GitHub-based libraries like FastBLE
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "hellobellohellobello"

// Centralized Android SDK detection function
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

// Detect Android SDK presence to conditionally include Android modules
val hasAndroidSdk = detectAndroidSdk()

if (hasAndroidSdk) {
    include(":android_sensor_node")
    project(":android_sensor_node").projectDir = file("android_sensor_node")
    include(":android_sensor_node:app")
    project(":android_sensor_node:app").projectDir = file("android_sensor_node/app")
    include(":android_sensor_node:libapp")
    project(":android_sensor_node:libapp").projectDir = file("android_sensor_node/libapp")
} else {
    println("[settings] Android SDK not found; skipping inclusion of :android_sensor_node modules")
}

include(":pc_controller")
project(":pc_controller").projectDir = file("pc_controller")
