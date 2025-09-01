plugins {
    id("com.android.library")
    id("kotlin-android")
    id("kotlin-android-extensions")
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
        minSdk = 24
        targetSdk = 34
        
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }
    
    buildTypes {
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
        dataBinding = true
        viewBinding = true
    }
}

dependencies {
    api(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
    
    // AndroidX
    api("androidx.appcompat:appcompat:1.6.1")
    api("androidx.core:core-ktx:1.12.0")
    api("androidx.fragment:fragment-ktx:1.6.2")
    api("com.google.android.material:material:1.11.0")
    
    // Lifecycle
    api("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    api("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    api("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    
    // Networking
    api("com.squareup.retrofit2:retrofit:2.9.0")
    api("com.squareup.retrofit2:converter-gson:2.9.0")
    api("com.squareup.okhttp3:logging-interceptor:4.12.0")
    
    // Image Loading
    api("com.github.bumptech.glide:glide:4.16.0")
    kapt("com.github.bumptech.glide:compiler:4.16.0")
    
    // Utilities
    api("org.greenrobot:eventbus:3.3.1")
    api("com.blankj:utilcode:1.30.7")
    
    // Permissions
    api("com.github.getActivity:XXPermissions:18.5")
    
    // Logging
    api("com.elvishew:xlog:1.11.1")
    
    // Camera
    api("androidx.camera:camera-core:1.3.1")
    api("androidx.camera:camera-camera2:1.3.1")
    api("androidx.camera:camera-lifecycle:1.3.1")
    api("androidx.camera:camera-view:1.3.1")
    
    // Test
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}