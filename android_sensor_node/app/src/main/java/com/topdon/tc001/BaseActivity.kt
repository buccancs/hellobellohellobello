package com.topdon.tc001

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

abstract class BaseActivity : AppCompatActivity() {

    protected var savedInstanceState: Bundle? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        this.savedInstanceState = savedInstanceState
        setContentView(initContentView())
        initView()
        initData()
    }

    /**
     * 初始化布局
     * @return 布局资源ID
     */
    abstract fun initContentView(): Int

    /**
     * 初始化视图
     */
    abstract fun initView()

    /**
     * 初始化数据
     */
    abstract fun initData()

    /**
     * 设备连接
     */
    open fun connected() {
        // Override in subclass if needed
    }

    /**
     * 设备断开连接
     */
    open fun disConnected() {
        // Override in subclass if needed
    }
}