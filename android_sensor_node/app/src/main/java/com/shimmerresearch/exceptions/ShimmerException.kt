package com.shimmerresearch.exceptions

/**
 * Shimmer-specific exception class from ShimmerAndroidAPI
 */
open class ShimmerException : kotlin.Exception {
    constructor() : super()

    constructor(message: String) : super(message)

    constructor(message: String, cause: Throwable) : super(message, cause)

    constructor(cause: Throwable) : super(cause)
}

/**
 * Connection-specific exception
 */
class ShimmerConnectionException : ShimmerException {
    constructor(message: String) : super(message)

    constructor(message: String, cause: Throwable) : super(message, cause)
}

/**
 * Configuration-specific exception
 */
class ShimmerConfigurationException : ShimmerException {
    constructor(message: String) : super(message)

    constructor(message: String, cause: Throwable) : super(message, cause)
}
