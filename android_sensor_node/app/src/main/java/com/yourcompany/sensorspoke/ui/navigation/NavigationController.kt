package com.yourcompany.sensorspoke.ui.navigation

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.widget.ViewPager2
import com.yourcompany.sensorspoke.ui.fragments.ThermalPreviewFragment

/**
 * NavigationController - Enhanced routing architecture inspired by IRCamera
 *
 * Provides centralized navigation management with:
 * - Fragment lifecycle management
 * - Tab-based routing with enhanced capabilities
 * - Deep linking support for thermal camera states
 * - Context-aware navigation transitions
 */
class NavigationController(
    private val activity: FragmentActivity,
    private val viewPager: ViewPager2,
) {
    companion object {
        const val TAB_RGB_CAMERA = 0
        const val TAB_THERMAL_CAMERA = 1
        const val TAB_TC001_MANAGEMENT = 2
        const val TAB_FILE_MANAGER = 3
        const val TAB_COUNT = 4

        const val ROUTE_RGB_PREVIEW = "rgb_preview"
        const val ROUTE_THERMAL_PREVIEW = "thermal_preview"
        const val ROUTE_THERMAL_SETTINGS = "thermal_settings"
        const val ROUTE_TC001_MANAGEMENT = "tc001_management"
        const val ROUTE_FILE_MANAGER = "file_manager"
    }

    private var currentTabIndex = TAB_RGB_CAMERA
    private val navigationHistory = mutableListOf<String>()

    /**
     * Navigate to specific tab with enhanced routing capabilities
     */
    fun navigateToTab(
        tabIndex: Int,
        route: String = "",
    ) {
        if (tabIndex in 0 until TAB_COUNT) {
            currentTabIndex = tabIndex
            viewPager.currentItem = tabIndex

            val routeName =
                when (tabIndex) {
                    TAB_RGB_CAMERA -> ROUTE_RGB_PREVIEW
                    TAB_THERMAL_CAMERA -> if (route.isNotEmpty()) route else ROUTE_THERMAL_PREVIEW
                    TAB_TC001_MANAGEMENT -> ROUTE_TC001_MANAGEMENT
                    TAB_FILE_MANAGER -> ROUTE_FILE_MANAGER
                    else -> "unknown"
                }

            addToNavigationHistory(routeName)
        }
    }

    /**
     * Navigate to TC001 management interface
     */
    fun navigateToTC001Management() {
        navigateToTab(TAB_TC001_MANAGEMENT, ROUTE_TC001_MANAGEMENT)
    }

    /**
     * Navigate to thermal camera with specific state/settings
     */
    fun navigateToThermalCamera(thermalState: ThermalNavigationState = ThermalNavigationState.PREVIEW) {
        navigateToTab(TAB_THERMAL_CAMERA, thermalState.route)

        val thermalFragment = getCurrentFragment() as? ThermalPreviewFragment
        thermalFragment?.handleNavigationState(thermalState)
    }

    /**
     * Get current active fragment
     */
    fun getCurrentFragment(): Fragment? {
        val adapter = viewPager.adapter as? com.yourcompany.sensorspoke.ui.adapters.MainPagerAdapter
        return adapter?.let {
            activity.supportFragmentManager.findFragmentByTag("f$currentTabIndex")
        }
    }

    /**
     * Check if we can navigate back
     */
    fun canNavigateBack(): Boolean = navigationHistory.size > 1

    /**
     * Navigate back in history
     */
    fun navigateBack(): Boolean {
        if (canNavigateBack()) {
            navigationHistory.removeLastOrNull()
            val previousRoute = navigationHistory.lastOrNull()

            previousRoute?.let { route ->
                when (route) {
                    ROUTE_RGB_PREVIEW -> navigateToTab(TAB_RGB_CAMERA)
                    ROUTE_THERMAL_PREVIEW -> navigateToTab(TAB_THERMAL_CAMERA)
                    ROUTE_TC001_MANAGEMENT -> navigateToTab(TAB_TC001_MANAGEMENT)
                    ROUTE_FILE_MANAGER -> navigateToTab(TAB_FILE_MANAGER)
                }
                return true
            }
        }
        return false
    }

    /**
     * Get current tab index
     */
    fun getCurrentTabIndex(): Int = currentTabIndex

    /**
     * Update current tab index (called by ViewPager listener)
     */
    fun updateCurrentTab(tabIndex: Int) {
        currentTabIndex = tabIndex
    }

    private fun addToNavigationHistory(route: String) {
        if (navigationHistory.isEmpty() || navigationHistory.last() != route) {
            navigationHistory.add(route)

            if (navigationHistory.size > 10) {
                navigationHistory.removeAt(0)
            }
        }
    }

    /**
     * Get navigation breadcrumbs for debugging/analytics
     */
    fun getNavigationBreadcrumbs(): List<String> = navigationHistory.toList()
}

/**
 * Enhanced thermal navigation states for better TC001 integration
 */
enum class ThermalNavigationState(
    val route: String,
) {
    PREVIEW("thermal_preview"),
    SETTINGS("thermal_settings"),
    CALIBRATION("thermal_calibration"),
    ANALYSIS("thermal_analysis"),
}
