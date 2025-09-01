package com.yourcompany.sensorspoke.ui.adapters

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter
import com.yourcompany.sensorspoke.ui.fragments.FileManagerFragment
import com.yourcompany.sensorspoke.ui.fragments.RgbPreviewFragment
import com.yourcompany.sensorspoke.ui.fragments.TC001ManagementFragment
import com.yourcompany.sensorspoke.ui.fragments.ThermalPreviewFragment

/**
 * Enhanced ViewPager2 adapter for main activity tabs with TC001 management
 */
class MainPagerAdapter(
    fragmentActivity: FragmentActivity,
) : FragmentStateAdapter(fragmentActivity) {
    companion object {
        const val TAB_RGB_PREVIEW = 0
        const val TAB_THERMAL_PREVIEW = 1
        const val TAB_TC001_MANAGEMENT = 2
        const val TAB_FILE_MANAGER = 3
        const val TAB_COUNT = 4

        val TAB_TITLES =
            arrayOf(
                "RGB Camera",
                "Thermal Camera",
                "TC001 Management",
                "Files",
            )
    }

    override fun getItemCount(): Int = TAB_COUNT

    override fun createFragment(position: Int): Fragment =
        when (position) {
            TAB_RGB_PREVIEW -> RgbPreviewFragment.newInstance()
            TAB_THERMAL_PREVIEW -> ThermalPreviewFragment.newInstance()
            TAB_TC001_MANAGEMENT -> TC001ManagementFragment.newInstance()
            TAB_FILE_MANAGER -> FileManagerFragment.newInstance()
            else -> throw IllegalArgumentException("Invalid tab position: $position")
        }
}
