package com.shimmerresearch.driver

import com.shimmerresearch.bluetooth.ShimmerBluetooth
import java.util.concurrent.ConcurrentHashMap

/**
 * ObjectCluster data container from ShimmerAndroidAPI.
 * Stores sensor data in multiple formats (RAW, CAL) with metadata.
 */
class ObjectCluster {
    var mState: ShimmerBluetooth.BtState = ShimmerBluetooth.BtState.DISCONNECTED
    private var mBluetoothAddress: String = ""

    private val mDataMap = ConcurrentHashMap<String, ConcurrentHashMap<String, FormatCluster>>()

    /**
     * Add sensor data in specified format
     */
    fun addData(
        sensorName: String,
        formatType: String,
        units: String,
        data: Double,
    ) {
        if (!mDataMap.containsKey(sensorName)) {
            mDataMap[sensorName] = ConcurrentHashMap()
        }

        val formatCluster = FormatCluster(formatType, units, data)
        mDataMap[sensorName]?.put(formatType, formatCluster)
    }

    /**
     * Get all format clusters for a sensor
     */
    fun getCollectionOfFormatClusters(sensorName: String): Collection<FormatCluster>? = mDataMap[sensorName]?.values

    /**
     * Get specific format cluster
     */
    fun getFormatCluster(
        sensorName: String,
        formatType: String,
    ): FormatCluster? = mDataMap[sensorName]?.get(formatType)

    /**
     * Return format cluster from collection by type
     */
    companion object {
        fun returnFormatCluster(
            formatClusters: Collection<FormatCluster>,
            formatType: String,
        ): FormatCluster? = formatClusters.find { it.mChannelType == formatType }
    }

    /**
     * Set Bluetooth address
     */
    fun setBluetoothAddress(address: String) {
        mBluetoothAddress = address
    }

    /**
     * Get Bluetooth address
     */
    fun getMacAddress(): String = mBluetoothAddress

    /**
     * Get all sensor names
     */
    fun getSensorNames(): Set<String> = mDataMap.keys

    /**
     * Check if sensor data exists
     */
    fun containsSensor(sensorName: String): Boolean = mDataMap.containsKey(sensorName)

    /**
     * Get formats for sensor
     */
    fun getFormatsForSensor(sensorName: String): Set<String>? = mDataMap[sensorName]?.keys

    override fun toString(): String {
        val sb = StringBuilder()
        sb.append("ObjectCluster[$mBluetoothAddress] - State: $mState\n")
        for ((sensor, formats) in mDataMap) {
            sb.append("  $sensor: ")
            for ((format, cluster) in formats) {
                sb.append("$format=${cluster.mData}${cluster.mUnits} ")
            }
            sb.append("\n")
        }
        return sb.toString()
    }
}

/**
 * FormatCluster contains sensor data in specific format
 */
data class FormatCluster(
    val mChannelType: String,
    val mUnits: String,
    val mData: Double,
)
