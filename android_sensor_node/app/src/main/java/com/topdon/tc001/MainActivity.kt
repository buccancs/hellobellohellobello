package com.topdon.tc001

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.util.SparseArray
import android.view.KeyEvent
import android.view.View
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.blankj.utilcode.util.AppUtils
import com.hjq.permissions.OnPermissionCallback
import com.hjq.permissions.Permission
import com.hjq.permissions.XXPermissions
import com.topdon.tc001.fragment.MainFragment
import com.topdon.tc001.databinding.ActivityMainBinding
import kotlinx.coroutines.launch


class MainActivity : BaseActivity(), View.OnClickListener {

    private lateinit var binding: ActivityMainBinding
    private var checkPermissionType: Int = -1 //0 initData数据 1 图库  2 connect方法
    
    override fun initContentView(): Int = R.layout.activity_main

    //记录设备信息
    private fun logInfo() {
        try {
            val str = StringBuilder()
            str.append("Info").append("\n")
            str.append("VERSION_CODE: ${BuildConfig.VERSION_CODE}").append("\n")
            str.append("VERSION_NAME: ${BuildConfig.VERSION_NAME}").append("\n")
            str.append("BRAND: ${Build.BRAND}").append("\n")
            str.append("MODEL: ${Build.MODEL}").append("\n")
            str.append("PRODUCT: ${Build.PRODUCT}").append("\n")
            str.append("CPU_ABI: ${Build.SUPPORTED_ABIS.joinToString(",")}").append("\n")
            str.append("SDK_INT: ${Build.VERSION.SDK_INT}").append("\n")
            str.append("RELEASE: ${Build.VERSION.RELEASE}").append("\n")
            println(str)
        } catch (e: Exception) {
            println("log error: ${e.message}")
        }
    }

    override fun initView() {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        logInfo()
        
        binding.viewPage.offscreenPageLimit = 3
        binding.viewPage.isUserInputEnabled = false
        binding.viewPage.adapter = ViewPagerAdapter(this)
        binding.viewPage.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                refreshTabSelect(position)
            }
        })
        if (savedInstanceState == null) {
            binding.viewPage.setCurrentItem(1, false)
        }

        binding.clIconGallery.setOnClickListener(this)
        binding.viewMain.setOnClickListener(this)
        binding.clIconMine.setOnClickListener(this)
    }

    override fun initData() {
        checkPermissionType = 0
        checkCameraPermission()
    }

    override fun onResume() {
        super.onResume()
    }

    override fun onPause() {
        super.onPause()
    }

    override fun onClick(v: View?) {
        when (v) {
            binding.clIconGallery -> {//图库
                checkPermissionType = 1
                checkStoragePermission()
            }
            binding.viewMain -> {//首页
                binding.viewPage.setCurrentItem(1, false)
            }
            binding.clIconMine -> {//我的
                binding.viewPage.setCurrentItem(2, false)
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK) {
            // Show exit dialog
            showExitDialog()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
    
    private fun showExitDialog() {
        // Simplified exit dialog
        finish()
    }

    /**
     * 刷新 3 个 tab 的选中状态
     * @param index 当前选中哪个 tab，`[0, 2]`
     */
    private fun refreshTabSelect(index: Int) {
        binding.ivIconGallery.isSelected = false
        binding.tvIconGallery.isSelected = false
        binding.ivIconMine.isSelected = false
        binding.tvIconMine.isSelected = false
        binding.ivBottomMainBg.setImageResource(R.drawable.ic_main_bg_not_select)

        when (index) {
            0 -> {//图库
                binding.ivIconGallery.isSelected = true
                binding.tvIconGallery.isSelected = true
            }
            1 -> {
                binding.ivBottomMainBg.setImageResource(R.drawable.ic_main_bg_select)
            }
            2 -> {//我的
                binding.ivIconMine.isSelected = true
                binding.tvIconMine.isSelected = true
            }
        }
    }

    override fun connected() {
        if (true) { // Simplified check
            checkPermissionType = 2
            checkCameraPermission()
        }
    }

    override fun disConnected() {
        // Handle disconnect
    }

    private class ViewPagerAdapter(activity: FragmentActivity) : FragmentStateAdapter(activity) {
        override fun getItemCount() = 3

        override fun createFragment(position: Int): Fragment {
            return when (position) {
                0 -> {
                    // Gallery fragment placeholder
                    MainFragment()
                }
                1 -> MainFragment()
                else -> MainFragment() // Mine fragment placeholder
            }
        }
    }

    /**
     * 权限检测
     */
    private fun getNeedPermissionList(): SparseArray<List<String>> {
        val sparseArray = SparseArray<List<String>>()
        sparseArray.append(R.string.permission_request_camera_app, listOf(Manifest.permission.CAMERA))
        
        (if (this.applicationInfo.targetSdkVersion >= 34){
            listOf(
                Permission.READ_MEDIA_VIDEO,
                Permission.READ_MEDIA_IMAGES,
                Permission.WRITE_EXTERNAL_STORAGE
            )
        } else if (this.applicationInfo.targetSdkVersion == 33) {
            listOf(
                Permission.READ_MEDIA_VIDEO,
                Permission.READ_MEDIA_IMAGES,
                Permission.WRITE_EXTERNAL_STORAGE
            )
        } else {
            listOf(Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE)
        }).let {
            sparseArray.append(R.string.permission_request_storage_app, it)
        }
        
        // BLE permissions for GSR sensor
        val blePermissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            listOf(
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_ADVERTISE
            )
        } else {
            listOf(
                Manifest.permission.BLUETOOTH,
                Manifest.permission.BLUETOOTH_ADMIN,
                Manifest.permission.ACCESS_FINE_LOCATION
            )
        }
        sparseArray.append(R.string.permission_request_ble_app, blePermissions)
        
        return sparseArray
    }

    private fun checkCameraPermission() {
        if (!XXPermissions.isGranted(
                this,
                getNeedPermissionList()[R.string.permission_request_camera_app]
            )
        ) {
            initCameraPermission()
        } else {
            initCameraPermission()
        }
    }

    /**
     * 动态申请权限
     */
    private fun initCameraPermission() {
        XXPermissions.with(this)
            .permission(getNeedPermissionList()[R.string.permission_request_camera_app])
            .request(object : OnPermissionCallback {
                override fun onGranted(permissions: MutableList<String>, allGranted: Boolean) {
                    if (allGranted) {
                        checkStoragePermission()
                    }
                }

                override fun onDenied(permissions: MutableList<String>, doNotAskAgain: Boolean) {
                    // Handle denial
                }
            })
    }

    private fun checkStoragePermission() {
        if (!XXPermissions.isGranted(this, getNeedPermissionList()[R.string.permission_request_storage_app])) {
            initStoragePermission()
        } else {
            initStoragePermission()
        }
    }

    /**
     * 动态申请权限
     */
    private fun initStoragePermission() {
        XXPermissions.with(this)
            .permission(
                getNeedPermissionList()[R.string.permission_request_storage_app]
            )
            .request(object : OnPermissionCallback {
                override fun onGranted(permissions: MutableList<String>, allGranted: Boolean) {
                    if (allGranted) {
                        jumpIRActivity()
                    }
                }

                override fun onDenied(permissions: MutableList<String>, doNotAskAgain: Boolean) {
                    // Handle denial
                }
            })
    }

    fun jumpIRActivity(){
        when (checkPermissionType) {
            0 -> {
                // Device connection check
            }
            1 -> {
                binding.viewPage.setCurrentItem(0, false)
            }
            2 -> {
                // Open thermal camera activity
            }
        }
    }
}