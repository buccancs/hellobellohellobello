plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jlleitschuh.gradle.ktlint") version "12.1.2"
}

android {
    namespace = "com.yourcompany.sensorspoke"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.yourcompany.sensorspoke"
        minSdk = 26 // Android 8.0 (API 26) - matches documentation
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        viewBinding = true
        // Performance: disable unused features
        buildConfig = false
        aidl = false
        renderScript = false
        shaders = false
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.isReturnDefaultValues = true
        // Performance: enable parallel test execution
        execution = "ANDROIDX_TEST_ORCHESTRATOR"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        // Set language version to support advanced features like break/continue in inline lambdas
        languageVersion = "1.9"
        apiVersion = "1.9"
        // Performance: enable incremental compilation
        freeCompilerArgs +=
            listOf(
                "-opt-in=kotlin.RequiresOptIn",
            )
    }

    packaging {
        jniLibs {
            pickFirsts +=
                setOf(
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
                )
        }
    }
}

dependencies {
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

    // CameraX for RGB recording - compatible versions with AGP 8.5.2
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-video:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // Camera2 API for advanced RAW capture capabilities
    implementation("androidx.camera:camera-extensions:$cameraxVersion")

    // Samsung Camera SDK for advanced features and RAW DNG capture
    // Note: Samsung Camera SDK would be added here when available
    // For now, we'll use Camera2 API with Samsung-specific optimizations

    // TLS networking, background work, and encryption - compatible versions
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.work:work-runtime-ktx:2.9.1")
    implementation("androidx.security:security-crypto:1.1.0")

    // Local SDKs (Topdon TC001 and Shimmer Android API)
    implementation(files("src/main/libs/topdon_1.3.7.aar"))
    implementation(files("src/main/libs/libusbdualsdk_1.3.4_2406271906_standard.aar"))
    implementation(files("src/main/libs/opengl_1.3.2_standard.aar"))
    implementation(files("src/main/libs/suplib-release.aar"))

    implementation(files("src/main/libs/shimmerandroidinstrumentdriver-3.2.3_beta.aar"))
    implementation(files("src/main/libs/shimmerbluetoothmanager-0.11.4_beta.jar"))
    implementation(files("src/main/libs/shimmerdriver-0.11.4_beta.jar"))
    implementation(files("src/main/libs/shimmerdriverpc-0.11.4_beta.jar"))

    // FastBLE for robust BLE communication with Shimmer devices
    implementation("com.github.Jasonchenlijian:FastBle:2.4.0")

    // Unit testing - compatible versions
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.13")
    testImplementation("androidx.test:core:1.6.1")
    testImplementation("androidx.test.ext:junit:1.2.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation("com.google.truth:truth:1.4.4")
    testImplementation("io.mockk:mockk:1.13.14")
    testImplementation("androidx.test:runner:1.6.2")

    // Performance: Test orchestrator for parallel execution
    androidTestUtil("androidx.test:orchestrator:1.5.1")
}

// Enable detailed per-test logging for JVM unit tests
// Note: Use fully-qualified names to avoid script-level imports.

tasks.withType<org.gradle.api.tasks.testing.Test>().configureEach {
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
