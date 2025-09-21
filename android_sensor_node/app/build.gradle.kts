plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jlleitschuh.gradle.ktlint") version "12.1.2"
}

android {
    namespace = "com.yourcompany.sensorspoke"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.yourcompany.sensorspoke"
        minSdk = 26 // Android 8.0 (API 26) - matches documentation
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // Common build configuration fields
        // Note: BUILD_TYPE is defined per build type to ensure correct values
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-DEBUG"
            isDebuggable = true
            isMinifyEnabled = false
            isShrinkResources = false
            
            buildConfigField("String", "BUILD_TYPE", "\"debug\"")
            buildConfigField("boolean", "DEBUG_MODE", "true")
            buildConfigField("String", "API_BASE_URL", "\"http://localhost:8080\"")
        }
        
        release {
            isDebuggable = false
            isMinifyEnabled = true
            isShrinkResources = true
            
            buildConfigField("String", "BUILD_TYPE", "\"release\"")
            buildConfigField("boolean", "DEBUG_MODE", "false")
            buildConfigField("String", "API_BASE_URL", "\"https://api.production.com\"")
            
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            
            // Signing config would go here for release builds
            // signingConfig = signingConfigs.getByName("release")
        }
        
        create("staging") {
            initWith(buildTypes.getByName("release"))
            applicationIdSuffix = ".staging"
            versionNameSuffix = "-STAGING"
            isDebuggable = true
            
            buildConfigField("String", "BUILD_TYPE", "\"staging\"")
            buildConfigField("boolean", "DEBUG_MODE", "true")
            buildConfigField("String", "API_BASE_URL", "\"https://staging-api.com\"")
        }
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true // Enable BuildConfig generation
        // Performance: disable unused features
        aidl = false
        renderScript = false
        shaders = false
    }
    
    // Product flavors for different hardware configurations
    flavorDimensions += "hardware"
    productFlavors {
        create("full") {
            dimension = "hardware"
            buildConfigField("boolean", "SUPPORTS_THERMAL_CAMERA", "true")
            buildConfigField("boolean", "SUPPORTS_SHIMMER", "true")
            buildConfigField("boolean", "SUPPORTS_RGB_CAMERA", "true")
        }
        
        create("lite") {
            dimension = "hardware"
            applicationIdSuffix = ".lite"
            buildConfigField("boolean", "SUPPORTS_THERMAL_CAMERA", "false")
            buildConfigField("boolean", "SUPPORTS_SHIMMER", "true")
            buildConfigField("boolean", "SUPPORTS_RGB_CAMERA", "true")
        }
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.isReturnDefaultValues = true
        // Performance: enable parallel test execution
        execution = "ANDROIDX_TEST_ORCHESTRATOR"
        
        // Remove JUnit Platform configuration as it's not compatible with Android
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    
    kotlinOptions {
        jvmTarget = "17"
        // Performance: enable incremental compilation
        freeCompilerArgs += listOf(
            "-opt-in=kotlin.RequiresOptIn",
            "-Xjsr305=strict"
            // Removed "-Xuse-k2" flag as it may cause compilation issues
        )
    }

    packaging {
        jniLibs {
            pickFirsts += setOf(
                "lib/**/libUSBUVCCamera.so",
                "lib/**/libencrypt.so",
                "lib/**/libircmd.so",
                "lib/**/libirparse.so",
                "lib/**/libirprocess.so",
                "lib/**/libirtemp.so",
                "lib/**/libomp.so",
                "lib/**/libirnet.so",
                "lib/**/libusb-1.0.so",
                "lib/**/libusbcamera.so",
                "lib/**/libopencv_java4.so", // Fix for conflicting OpenCV libraries
            )
        }
        resources {
            excludes += setOf(
                "META-INF/DEPENDENCIES",
                "META-INF/LICENSE",
                "META-INF/LICENSE.txt",
                "META-INF/NOTICE",
                "META-INF/NOTICE.txt",
                "META-INF/*.kotlin_module",
                "META-INF/*_release.kotlin_module"
            )
        }
    }
    
    // Lint configuration
    lint {
        abortOnError = System.getenv("CI") == "true"
        checkReleaseBuilds = true
        disable += setOf("MissingTranslation", "UnusedResources")
    }
}

dependencies {
    // ============================================================================
    // Core Android Dependencies
    // ============================================================================
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.fragment:fragment-ktx:1.8.5")
    implementation("androidx.viewpager2:viewpager2:1.1.0")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // ============================================================================
    // Camera and Media Capture
    // ============================================================================
    // CameraX for RGB recording - compatible versions with AGP 8.1.4
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-video:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    implementation("androidx.camera:camera-extensions:$cameraxVersion") // Camera2 API for advanced RAW capture

    // ============================================================================
    // Networking and Security
    // ============================================================================
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.work:work-runtime-ktx:2.9.1")
    implementation("androidx.security:security-crypto:1.1.0")

    // ============================================================================
    // UI and Animation Libraries
    // ============================================================================
    implementation("com.airbnb.android:lottie:6.1.0") // Required by Shimmer SDK and animations

    // ============================================================================
    // Hardware SDK Dependencies
    // ============================================================================
    // Local SDKs (Topdon TC001 and Shimmer Android API) - from libs directory
    implementation(files("libs/topdon.aar"))
    implementation(files("libs/libAC020sdk_USB_IR_1.1.1_2408291439.aar"))
    implementation(files("libs/libusbdualsdk_1.3.4_2406271906_standard.aar"))
    implementation(files("libs/opengl_1.3.2_standard.aar"))
    implementation(files("libs/suplib-release.aar"))
    implementation(files("libs/libcommon_1.2.0_24052117.aar"))
    implementation(files("libs/libirutils_1.2.0_2409241055.aar"))
    implementation(files("libs/abtest-1.0.1.aar"))
    implementation(files("libs/auth-number-2.13.2.1.aar"))
    implementation(files("libs/logger-2.2.1-release.aar"))
    implementation(files("libs/main-2.2.1-release.aar"))

    // FastBLE for robust BLE communication with Shimmer devices
    implementation("com.github.Jasonchenlijian:FastBle:2.4.0")

    // ============================================================================
    // Testing Dependencies
    // ============================================================================
    // Unit Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.13")
    testImplementation("androidx.test:core:1.6.1")
    testImplementation("androidx.test.ext:junit:1.2.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation("com.google.truth:truth:1.4.4")
    testImplementation("io.mockk:mockk:1.13.14")
    testImplementation("androidx.test:runner:1.6.2")

    // Instrumented Testing
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    
    // Performance: Test orchestrator for parallel execution
    androidTestUtil("androidx.test:orchestrator:1.5.1")
}

// ============================================================================
// Task Configuration
// ============================================================================

// Enable detailed per-test logging for JVM unit tests
tasks.withType<Test>().configureEach {
    testLogging {
        events(
            org.gradle.api.tasks.testing.logging.TestLogEvent.STARTED,
            org.gradle.api.tasks.testing.logging.TestLogEvent.PASSED,
            org.gradle.api.tasks.testing.logging.TestLogEvent.SKIPPED,
            org.gradle.api.tasks.testing.logging.TestLogEvent.FAILED,
            org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_OUT,
            org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_ERROR,
        )
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}

// Ktlint configuration - stable version
ktlint {
    version.set("0.50.0")
    android.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    verbose.set(true)
    // Performance: disable slow rules, modern ktlint approach
    filter {
        exclude("**/generated/**")
        include("**/kotlin/**")
    }
}

// ============================================================================
// Custom Build Tasks
// ============================================================================

tasks.register("assembleAllVariants") {
    group = "build"
    description = "Assemble all build variants for comprehensive testing"
    dependsOn(
        "assembleFullDebug",
        "assembleFullRelease",
        "assembleFullStaging",
        "assembleLiteDebug",
        "assembleLiteRelease",
        "assembleLiteStaging"
    )
}

tasks.register("testAllVariants") {
    group = "verification"  
    description = "Run unit tests for all debug build variants"
    dependsOn(
        "testFullDebugUnitTest",
        "testLiteDebugUnitTest"
    )
}
