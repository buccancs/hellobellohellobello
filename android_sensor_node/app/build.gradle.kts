plugins {
    id("com.android.application")
    id("kotlin-android")
    id("kotlin-kapt")
}

kapt {
    arguments {
        arg("AROUTER_MODULE_NAME", project.name)
    }
}

android {
    compileSdk = 34
    
    defaultConfig {
        applicationId = "com.topdon.tc001"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        multiDexEnabled = true
        
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        
        buildConfigField("String", "VERSION_DATE", "\"${System.currentTimeMillis()}\"")
        
        setProperty("archivesBaseName", "IRCamera-v${versionName}")
    }
    
    signingConfigs {
        create("release") {
            // Add signing config if needed
        }
    }
    
    buildTypes {
        debug {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android.txt"),
                "proguard-rules.pro"
            )
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    
    kotlinOptions {
        jvmTarget = "1.8"
    }
    
    buildFeatures {
        buildConfig = true
        dataBinding = true
        viewBinding = true
    }
    
    packaging {
        resources {
            pickFirsts += listOf(
                "lib/x86/libc++_shared.so",
                "lib/x86_64/libc++_shared.so",
                "lib/arm64-v8a/libc++_shared.so",
                "lib/armeabi-v7a/libc++_shared.so"
            )
        }
    }
    
    namespace = "com.topdon.tc001"
}

dependencies {
    // Local library module
    implementation(project(":android_sensor_node:libapp"))
    
    // AndroidX
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.viewpager2:viewpager2:1.0.0")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    implementation("androidx.fragment:fragment-ktx:1.6.2")
    
    // Third-party libraries (simplified for initial migration)
    implementation("com.github.bumptech.glide:glide:4.16.0")
    kapt("com.github.bumptech.glide:compiler:4.16.0")
    
    // Utilities
    implementation("com.blankj:utilcode:1.30.7")
    
    // Permissions
    implementation("com.github.getActivity:XXPermissions:18.5")
    
    // Camera
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")
    
    // GSR Sensor Integration - Enhanced Nordic BLE implementation
    // Implementing ShimmerAndroidAPI-compatible interface using robust BLE library
    implementation("no.nordicsemi.android:ble:2.6.1")
    implementation("no.nordicsemi.android:ble-ktx:2.6.1")
    
    // Additional dependencies for enhanced GSR support
    implementation("com.google.guava:guava:31.1-android")
    
    // CSV file handling for GSR data logging
    implementation("com.opencsv:opencsv:5.7.1")
    
    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // Test dependencies
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}