package com.yourcompany.sensorspoke.ui.popup

import android.content.Context
import android.view.View
import android.widget.PopupWindow
import android.widget.TextView
import androidx.constraintlayout.widget.ConstraintLayout
import com.yourcompany.sensorspoke.R

/**
 * DelPopup - Device deletion popup inspired by IRCamera
 *
 * Provides intuitive device deletion interface with confirmation
 */
class DelPopup(
    private val context: Context,
) : PopupWindow() {
    var onDelListener: (() -> Unit)? = null

    init {
        setupPopup()
    }

    private fun setupPopup() {
        // Create popup content programmatically since we don't have the exact layout
        val layout =
            ConstraintLayout(context).apply {
                layoutParams =
                    ConstraintLayout.LayoutParams(
                        ConstraintLayout.LayoutParams.WRAP_CONTENT,
                        ConstraintLayout.LayoutParams.WRAP_CONTENT,
                    )
                setBackgroundResource(R.drawable.svg_popup_del_bg)
                setPadding(32, 16, 32, 16)
            }

        val deleteText =
            TextView(context).apply {
                text = "Delete Device"
                textSize = 16f
                setTextColor(context.getColor(android.R.color.white))
                setPadding(16, 12, 16, 12)
                setOnClickListener {
                    onDelListener?.invoke()
                    dismiss()
                }
            }

        layout.addView(deleteText)

        contentView = layout
        width = ConstraintLayout.LayoutParams.WRAP_CONTENT
        height = ConstraintLayout.LayoutParams.WRAP_CONTENT
        isOutsideTouchable = true
        isFocusable = true
    }

    fun show(anchorView: View) {
        showAsDropDown(anchorView, 0, -anchorView.height)
    }
}
