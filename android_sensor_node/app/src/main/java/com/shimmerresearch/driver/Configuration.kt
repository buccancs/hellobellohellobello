package com.shimmerresearch.driver

/**
 * Configuration constants and sensor definitions from ShimmerAndroidAPI
 */
object Configuration {
    object Shimmer3 {
        /**
         * Object cluster sensor names
         */
        object ObjectClusterSensorName {
            const val TIMESTAMP = "TimeStamp"
            const val ACCEL_LN_X = "Low Noise Accelerometer X"
            const val ACCEL_LN_Y = "Low Noise Accelerometer Y"
            const val ACCEL_LN_Z = "Low Noise Accelerometer Z"
            const val GSR = "GSR"
            const val GSR_CONDUCTANCE = "GSR Conductance"
            const val PPG_A13 = "PPG A13"
            const val PPG = "PPG"
        }

        /**
         * Channel types for data formats
         */
        enum class ChannelType {
            RAW,
            CAL,
            UNCAL,
        }

        /**
         * Sensor bit flags
         */
        object SENSOR {
            const val ACCEL = 0x80L
            const val GYRO = 0x40L
            const val MAG = 0x20L
            const val GSR = 0x04L
            const val PPG_A13 = 0x08L
            const val TIMESTAMP = 0x01L
        }

        /**
         * GSR range settings
         */
        object GsrRange {
            const val AUTO = 0
            const val RANGE_4_7M = 1
            const val RANGE_2_3M = 2
            const val RANGE_1_2M = 3
            const val RANGE_560K = 4
        }

        /**
         * Sampling rates supported by Shimmer3
         */
        object SamplingRate {
            const val RATE_1HZ = 1.0
            const val RATE_10HZ = 10.24
            const val RATE_51HZ = 51.2
            const val RATE_102HZ = 102.4
            const val RATE_128HZ = 128.0
            const val RATE_170HZ = 170.67
            const val RATE_256HZ = 256.0
            const val RATE_512HZ = 512.0
            const val RATE_1024HZ = 1024.0
        }

        /**
         * Communication types
         */
        enum class CommunicationType {
            BLUETOOTH,
            BLUETOOTH_LE,
            DOCK,
        }
    }

    object Verisense {
        /**
         * Verisense sensor configurations
         */
        object SENSOR {
            const val ACCEL = 0x01L
            const val GYRO = 0x02L
            const val MAG = 0x04L
            const val PPG = 0x08L
            const val GSR = 0x10L
        }
    }
}
