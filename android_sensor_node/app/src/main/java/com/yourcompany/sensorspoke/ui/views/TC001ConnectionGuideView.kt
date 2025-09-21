package com.yourcompany.sensorspoke.ui.views

import android.content.Context
import android.content.res.TypedArray
import android.text.SpannableString
import android.text.Spanned
import android.text.TextUtils
import android.text.method.LinkMovementMethod
import android.text.style.ImageSpan
import android.util.AttributeSet
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.yourcompany.sensorspoke.R

/**
 * ConnectionGuideView - TC001 connection guidance UI
 *
 * Provides visual guidance for TC001 device connection
 * adapted from IRCamera's ConnectionGuideView
 */
class TC001ConnectionGuideView
@JvmOverloads
constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
) : LinearLayout(context, attrs, defStyleAttr) {
    private var iconRes: Int = 0
    private var contentStr: String = ""
    private var iconShow: Boolean = false
    private lateinit var guideIcon: ImageView
    private lateinit var contentText: TextView

    init {
        attrs?.let {
            val ta: TypedArray = context.obtainStyledAttributes(attrs, R.styleable.TC001ConnectionGuideView)
            iconRes = ta.getResourceId(R.styleable.TC001ConnectionGuideView_guide_icon, 0)
            contentStr = ta.getString(R.styleable.TC001ConnectionGuideView_guide_text) ?: ""
            iconShow = ta.getBoolean(R.styleable.TC001ConnectionGuideView_guide_icon_show, false)
            ta.recycle()
        }
        initView()
    }

    private fun initView() {
        inflate(context, R.layout.ui_tc001_connection_guide, this)
        contentText = findViewById(R.id.tv_content)
        guideIcon = findViewById(R.id.iv_icon)

        if (iconRes != 0) {
            guideIcon.setImageResource(iconRes)
        }
        contentText.text = contentStr
        guideIcon.visibility = if (iconShow) View.VISIBLE else View.GONE
    }

    fun setText(text: CharSequence?) {
        if (::contentText.isInitialized && !TextUtils.isEmpty(text)) {
            contentText.text = text
            contentText.movementMethod = LinkMovementMethod.getInstance()
        }
    }

    fun getText(): String =
        if (::contentText.isInitialized) {
            contentText.text.toString()
        } else {
            ""
        }

    fun setHighlightColor(color: Int) {
        if (::contentText.isInitialized) {
            contentText.highlightColor = color
        }
    }

    fun addConnectionIcon(content: String) {
        if (!::contentText.isInitialized) return

        var mContent = "$content  "
        val spannableString = SpannableString(mContent)

        val drawable = androidx.appcompat.content.res.AppCompatResources.getDrawable(context, R.drawable.ic_right_arrow)
        drawable?.setBounds(0, 0, drawable.minimumWidth, drawable.minimumHeight)
        spannableString.setSpan(
            ImageSpan(drawable!!),
            mContent.length - 1,
            mContent.length,
            Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
        )
        contentText.text = spannableString
    }
}
