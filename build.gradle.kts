plugins {
    base
}

// Project metadata and versioning
allprojects {
    group = "org.hellobellohellobello"
    version = "1.0-SNAPSHOT"
}

// ============================================================================
// Android SDK Detection and Environment Setup
// ============================================================================

/**
 * Detect whether an Android SDK is available on this machine
 */
fun detectAndroidSdk(): Boolean {
    // Check local.properties
    val localPropsFile = rootProject.file("local.properties")
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

val hasAndroidSdk = detectAndroidSdk()

// ============================================================================
// Composite Build Tasks
// ============================================================================

/**
 * Build all modules using only release variants - optimized for CI/CD
 */
tasks.register("buildRelease") {
    group = "build"
    description = "Builds all modules using only release variants for production deployment"
    
    if (hasAndroidSdk) {
        dependsOn(
            ":android_sensor_node:app:assembleFullRelease",
            ":android_sensor_node:app:assembleLiteRelease"
        )
    }
    dependsOn(":pc_controller:assemblePcController")
    
    doFirst {
        val androidStatus = if (hasAndroidSdk) "✓ Android SDK found" else "✗ Android SDK not found"
        println("[buildRelease] $androidStatus")
        println("[buildRelease] Building release variants for production deployment...")
    }
    
    doLast {
        println("[buildRelease] ✓ Production build completed successfully")
    }
}

/**
 * Clean all build artifacts across modules
 */
tasks.register<Delete>("cleanAll") {
    group = "build"
    description = "Clean all build artifacts and caches"
    
    delete(rootProject.layout.buildDirectory.get().asFile)
    if (hasAndroidSdk) {
        delete(project(":android_sensor_node:app").layout.buildDirectory.get().asFile)
    }
    delete(project(":pc_controller").layout.buildDirectory.get().asFile)
    
    doFirst {
        println("[cleanAll] Cleaning all build artifacts...")
    }
}

// ============================================================================
// Testing and Verification
// ============================================================================

/**
 * Run Python pytest suite
 */
tasks.register<Exec>("pyTest") {
    group = "verification"
    description = "Run Python pytest suite as defined by pytest.ini"
    
    executable = "python3"
    args = listOf("-m", "pytest", "--verbose", "--tb=short")
    workingDir = rootProject.projectDir
    
    doFirst {
        println("[pyTest] Running Python test suite...")
    }
}

/**
 * Comprehensive testing across all modules
 */
tasks.register("checkAll") {
    group = "verification"
    description = if (hasAndroidSdk) {
        "Run comprehensive tests: Android unit tests, Python pytest, and linting"
    } else {
        "Run Python pytest and linting (Android SDK not found; skipping Android tests)"
    }
    
    dependsOn("pyTest")
    
    if (hasAndroidSdk) {
        dependsOn(
            ":android_sensor_node:app:testFullDebugUnitTest",
            ":android_sensor_node:app:testLiteDebugUnitTest",
            ":android_sensor_node:app:lintFullDebug",
            ":android_sensor_node:app:lintLiteDebug"
        )
    }
    
    doFirst {
        val testScope = if (hasAndroidSdk) "Android + Python" else "Python only"
        println("[checkAll] Running comprehensive test suite: $testScope")
    }
    
    doLast {
        println("[checkAll] ✓ All verification tasks completed")
    }
}

// ============================================================================
// Packaging and Distribution
// ============================================================================

/**
 * Package all deployable artifacts
 */
tasks.register("packageAll") {
    group = "build"
    description = if (hasAndroidSdk) {
        "Package PC Controller executable and Android APK for distribution"
    } else {
        "Package PC Controller executable only (Android SDK not found)"
    }
    
    dependsOn(":pc_controller:assemblePcController")
    if (hasAndroidSdk) {
        dependsOn(
            ":android_sensor_node:app:assembleFullRelease",
            ":android_sensor_node:app:assembleLiteRelease"
        )
    }
    
    doFirst {
        println("[packageAll] Creating distribution packages...")
    }
    
    doLast {
        println("[packageAll] ✓ Distribution packages ready")
    }
}

/**
 * Generate comprehensive project report
 */
tasks.register("projectReport") {
    group = "help"
    description = "Generate comprehensive project build report"
    
    doLast {
        val androidStatus = if (detectAndroidSdk()) "✓ Available" else "✗ Not Found"
        println("""
        |================================================================================
        |Multi-Modal Physiological Sensing Platform - Build Configuration Report
        |================================================================================
        |Android SDK Status: $androidStatus
        |Gradle Version: ${gradle.gradleVersion}
        |Java Version: ${System.getProperty("java.version")}
        |Project Version: ${project.version}
        |
        |Available Build Tasks:
        | • ./gradlew buildRelease    - Build production release variants
        | • ./gradlew checkAll       - Run comprehensive test suite  
        | • ./gradlew packageAll     - Create distribution packages
        | • ./gradlew cleanAll       - Clean all build artifacts
        |
        |Module Status:
        | • PC Controller: ✓ Always available (Python)
        | • Android Sensor Node: $androidStatus
        |================================================================================
        """.trimMargin())
    }
}

// ============================================================================
// IDE and Tool Compatibility
// ============================================================================

/**
 * Placeholder classes task to satisfy IDEs/tools that invoke :classes at the root
 */
tasks.register("classes") {
    group = "build"
    description = "No-op placeholder for root project; use module-specific tasks instead"
    
    doFirst {
        println("[classes] Root project has no Java/Kotlin classes. Use module-specific tasks:")
        println("  • Android: ./gradlew :android_sensor_node:app:assembleDebug")
        println("  • Python: ./gradlew :pc_controller:assemblePcController")
    }
}

/**
 * Placeholder testClasses task to satisfy tools that invoke :testClasses at the root
 */
tasks.register("testClasses") {
    group = "verification"
    description = "No-op placeholder for root project; use module-specific tasks instead"
    
    doFirst {
        println("[testClasses] Root project has no test classes. Use module-specific tasks:")
        println("  • Android Tests: ./gradlew :android_sensor_node:app:testDebugUnitTest")
        println("  • Python Tests: ./gradlew pyTest")
    }
}

// ============================================================================
// Development and Debugging
// ============================================================================

/**
 * Development build with debug optimizations
 */
tasks.register("devBuild") {
    group = "build"
    description = "Development build with debug information and fast incremental compilation"
    
    if (hasAndroidSdk) {
        dependsOn(":android_sensor_node:app:assembleDebug")
    }
    dependsOn(":pc_controller:assemblePcController")
    
    doFirst {
        println("[devBuild] Creating development build with debug optimizations...")
    }
}
