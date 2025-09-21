package com.yourcompany.sensorspoke

import org.junit.Assert.fail
import org.junit.Assume
import org.junit.Test
import java.io.File
import java.util.concurrent.TimeUnit

class PytestBridgeTest {
    private fun findRepoRoot(start: File): File? {
        var cur: File? = start
        var steps = 0
        while (cur != null && steps < 12) {
            val pytestIni = File(cur, "pytest.ini")
            val pcTests = File(cur, "pc_controller/tests")
            if (pytestIni.exists() && pcTests.exists()) {
                return cur
            }
            cur = cur.parentFile
            steps++
        }
        return null
    }

    @Test
    fun runPythonPytestSuite() {
        val userDir = System.getProperty("user.dir")
        val start = File(userDir ?: ".").absoluteFile

        val repoRoot =
            findRepoRoot(start) ?: run {
                val candidate1 = start.parentFile?.parentFile
                val try1 = candidate1?.let { findRepoRoot(it) }
                if (try1 != null) {
                    try1
                } else {
                    Assume.assumeTrue("Repository root not found to run pytest", false)
                    return
                }
            }

        val cmd =
            arrayOf(
                "python3",
                "-m",
                "pytest",
                "-q",
                "pc_controller/tests",
            )
        val pb =
            ProcessBuilder(*cmd)
                .directory(repoRoot)
                .redirectErrorStream(true)

        val checkCmd = arrayOf("python3", "-m", "pytest", "--version")
        val checkPb =
            ProcessBuilder(*checkCmd)
                .directory(repoRoot)
                .redirectErrorStream(true)

        try {
            val checkProc = checkPb.start()
            val checkFinished = checkProc.waitFor(10, TimeUnit.SECONDS)
            if (!checkFinished || checkProc.exitValue() != 0) {
                Assume.assumeTrue("pytest module not available", false)
                return
            }
        } catch (e: Exception) {
            Assume.assumeTrue("python3 not available: ${e.message}", false)
            return
        }

        val proc =
            try {
                pb.start()
            } catch (e: Exception) {
                Assume.assumeTrue("python3 not available: ${e.message}", false)
                return
            }

        val finished = proc.waitFor(600, TimeUnit.SECONDS)
        val output = proc.inputStream.bufferedReader().readText()

        if (!finished) {
            proc.destroyForcibly()
            fail("pytest timed out. Output so far:\n$output")
        }

        val code = proc.exitValue()
        if (code != 0) {
            fail("pytest failed with exit code $code. Output:\n$output")
        }
    }
}
