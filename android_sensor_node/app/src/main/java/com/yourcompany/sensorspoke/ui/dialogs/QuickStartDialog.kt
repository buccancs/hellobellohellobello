package com.yourcompany.sensorspoke.ui.dialogs

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.yourcompany.sensorspoke.utils.UserExperience

/**
 * QuickStartDialog provides an interactive onboarding experience for new users
 * of the Android Sensor Node, guiding them through setup and connection.
 *
 * This complements the PC Controller's tutorial system to ensure consistent
 * user experience across both platforms.
 */
class QuickStartDialog(
    context: Context,
    private val onComplete: () -> Unit,
) : Dialog(context) {
    private var currentStep = 0
    private val steps = UserExperience.QuickStart.getSetupSteps()

    private lateinit var titleText: TextView
    private lateinit var stepText: TextView
    private lateinit var descriptionText: TextView
    private lateinit var prevButton: Button
    private lateinit var nextButton: Button
    private lateinit var finishButton: Button
    private lateinit var stepIndicators: LinearLayout

    init {
        setupDialog()
        updateStep()
    }

    private fun setupDialog() {
        val layout =
            LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(48, 48, 48, 48)
            }

        titleText =
            TextView(context).apply {
                text = "Quick Start Guide"
                textSize = 24f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 0, 0, 32)
            }
        layout.addView(titleText)

        stepIndicators =
            LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, 0, 0, 24)
            }

        for (i in steps.indices) {
            val dot =
                View(context).apply {
                    layoutParams =
                        LinearLayout.LayoutParams(24, 24).apply {
                            marginEnd = 16
                        }
                    background = ColorDrawable(if (i == 0) Color.parseColor("#2196F3") else Color.parseColor("#E0E0E0"))
                }
            stepIndicators.addView(dot)
        }
        layout.addView(stepIndicators)

        stepText =
            TextView(context).apply {
                textSize = 18f
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(0, 0, 0, 16)
            }
        layout.addView(stepText)

        descriptionText =
            TextView(context).apply {
                textSize = 16f
                setPadding(0, 0, 0, 32)
            }
        layout.addView(descriptionText)

        val buttonLayout =
            LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
            }

        prevButton =
            Button(context).apply {
                text = "Previous"
                isEnabled = false
                layoutParams =
                    LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginEnd = 8
                    }
                setOnClickListener { previousStep() }
            }
        buttonLayout.addView(prevButton)

        nextButton =
            Button(context).apply {
                text = "Next"
                layoutParams =
                    LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginStart = 8
                    }
                setOnClickListener { nextStep() }
            }
        buttonLayout.addView(nextButton)

        finishButton =
            Button(context).apply {
                text = "Get Started"
                visibility = View.GONE
                layoutParams =
                    LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        marginStart = 8
                    }
                setOnClickListener {
                    dismiss()
                    onComplete()
                }
            }
        buttonLayout.addView(finishButton)

        layout.addView(buttonLayout)

        setContentView(layout)

        // Dialog configuration
        window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        setCancelable(true)
        setCanceledOnTouchOutside(false)
    }

    private fun updateStep() {
        if (currentStep < steps.size) {
            val (stepTitle, stepDescription) = steps[currentStep]
            stepText.text = stepTitle
            descriptionText.text = stepDescription

            for (i in 0 until stepIndicators.childCount) {
                val dot = stepIndicators.getChildAt(i)
                dot.background =
                    ColorDrawable(
                        if (i <= currentStep) Color.parseColor("#2196F3") else Color.parseColor("#E0E0E0"),
                    )
            }

            prevButton.isEnabled = currentStep > 0

            if (currentStep == steps.size - 1) {
                nextButton.visibility = View.GONE
                finishButton.visibility = View.VISIBLE
            } else {
                nextButton.visibility = View.VISIBLE
                finishButton.visibility = View.GONE
            }
        }
    }

    private fun nextStep() {
        if (currentStep < steps.size - 1) {
            currentStep++
            updateStep()
        }
    }

    private fun previousStep() {
        if (currentStep > 0) {
            currentStep--
            updateStep()
        }
    }

    companion object {
        /**
         * Shows the quick start dialog with a completion callback.
         */
        fun show(
            context: Context,
            onComplete: () -> Unit,
        ) {
            QuickStartDialog(context, onComplete).show()
        }
    }
}
