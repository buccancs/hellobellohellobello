package com.yourcompany.sensorspoke.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.ui.adapters.SessionAdapter
import com.yourcompany.sensorspoke.ui.models.SessionInfo
import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/**
 * Fragment for managing recorded session files
 */
class FileManagerFragment : Fragment() {
    private var sessionsRecyclerView: RecyclerView? = null
    private var sessionCountText: TextView? = null
    private var btnRefresh: Button? = null
    private var btnCleanup: Button? = null

    private var sessionAdapter: SessionAdapter? = null
    private val sessionList = mutableListOf<SessionInfo>()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? = inflater.inflate(R.layout.fragment_file_manager, container, false)

    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?,
    ) {
        super.onViewCreated(view, savedInstanceState)

        sessionsRecyclerView = view.findViewById(R.id.sessionsRecyclerView)
        sessionCountText = view.findViewById(R.id.sessionCountText)
        btnRefresh = view.findViewById(R.id.btnRefresh)
        btnCleanup = view.findViewById(R.id.btnCleanup)

        setupRecyclerView()
        setupButtons()
        loadSessions()
    }

    private fun setupRecyclerView() {
        sessionAdapter =
            SessionAdapter(sessionList) { session ->
                onSessionClicked(session)
            }

        sessionsRecyclerView?.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = sessionAdapter
        }
    }

    private fun setupButtons() {
        btnRefresh?.setOnClickListener {
            loadSessions()
        }

        btnCleanup?.setOnClickListener {
            cleanupOldSessions()
        }
    }

    private fun loadSessions() {
        sessionList.clear()

        try {
            val context = requireContext()
            val sessionsDir = File(context.getExternalFilesDir(null), "sessions")

            if (!sessionsDir.exists()) {
                sessionsDir.mkdirs()
            }

            val sessionDirs =
                sessionsDir.listFiles { file ->
                    file.isDirectory && file.name.startsWith("session_")
                }

            sessionDirs?.forEach { sessionDir ->
                val sessionInfo = createSessionInfo(sessionDir)
                sessionList.add(sessionInfo)
            }

            sessionList.sortByDescending { it.dateTime }

            sessionAdapter?.notifyDataSetChanged()
            updateSessionCount()
        } catch (e: Exception) {
            Toast.makeText(requireContext(), "Error loading sessions: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun createSessionInfo(sessionDir: File): SessionInfo {
        val sessionName = sessionDir.name
        val dateTime = getSessionDateTime(sessionName)
        val size = calculateDirectorySize(sessionDir)

        val allFiles = sessionDir.walkTopDown().filter { it.isFile }.toList()
        val videoFiles = allFiles.count { it.name.endsWith(".mp4") }
        val csvFiles = allFiles.count { it.name.endsWith(".csv") }
        val imageFiles = allFiles.count { it.name.endsWith(".jpg") || it.name.endsWith(".jpeg") }

        val details =
            buildString {
                if (videoFiles > 0) append("$videoFiles video, ")
                if (csvFiles > 0) append("$csvFiles CSV, ")
                if (imageFiles > 0) append("$imageFiles images, ")
                append("${allFiles.size} total files")
            }

        return SessionInfo(
            name = sessionName,
            dateTime = dateTime,
            sizeBytes = size,
            details = details,
            directory = sessionDir,
        )
    }

    private fun getSessionDateTime(sessionName: String): Date {
        val pattern = """session_(\d{8})_(\d{6})""".toRegex()
        val matchResult = pattern.find(sessionName)

        return if (matchResult != null) {
            val (dateStr, timeStr) = matchResult.destructured
            val dateTimeStr = "${dateStr}_$timeStr"
            val format = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            format.parse(dateTimeStr) ?: Date()
        } else {
            Date()
        }
    }

    private fun calculateDirectorySize(directory: File): Long =
        try {
            directory
                .walkTopDown()
                .filter { it.isFile }
                .map { it.length() }
                .sum()
        } catch (e: Exception) {
            0L
        }

    private fun updateSessionCount() {
        val count = sessionList.size
        val totalSize = sessionList.sumOf { it.sizeBytes }
        val totalSizeMB = totalSize / (1024 * 1024)

        sessionCountText?.text = "$count sessions ($totalSizeMB MB)"
    }

    private fun onSessionClicked(session: SessionInfo) {
        Toast
            .makeText(
                requireContext(),
                "Session: ${session.name}\nSize: ${session.sizeBytes / (1024 * 1024)} MB",
                Toast.LENGTH_LONG,
            ).show()
    }

    private fun cleanupOldSessions() {
        try {
            // Enhanced cleanup with multiple options
            // Option 1: Age-based cleanup (configurable - default 30 days as mentioned in problem statement)
            val defaultCleanupDays = 30
            val cutoffDate = Calendar.getInstance().apply {
                add(Calendar.DAY_OF_MONTH, -defaultCleanupDays)
            }.time

            val oldSessions = sessionList.filter { it.dateTime.before(cutoffDate) }

            // Option 2: Size-based cleanup if storage is low
            val context = requireContext()
            val sessionsDir = File(context.getExternalFilesDir(null), "sessions")
            val statsFs = android.os.StatFs(sessionsDir.path)
            val availableBytes = statsFs.availableBytes
            val totalBytes = statsFs.totalBytes
            val usagePercent = ((totalBytes - availableBytes).toFloat() / totalBytes * 100f).toInt()

            val shouldCleanupBySize = usagePercent > 80
            val sizeMB = sessionList.sumOf { it.sizeBytes } / (1024 * 1024)

            android.util.Log.i("FileManager", "Cleanup analysis - Storage: $usagePercent% used, Sessions: ${sessionList.size} (${sizeMB}MB), Old sessions: ${oldSessions.size}")

            if (oldSessions.isEmpty() && !shouldCleanupBySize) {
                Toast.makeText(
                    requireContext(),
                    "No cleanup needed. Sessions: ${sessionList.size}, Storage: $usagePercent% used",
                    Toast.LENGTH_LONG,
                ).show()
                return
            }

            val sessionsToDelete = mutableListOf<SessionInfo>()

            sessionsToDelete.addAll(oldSessions)

            // If storage is still critical, add more sessions starting with largest/oldest
            if (shouldCleanupBySize && sessionsToDelete.size < sessionList.size) {
                val remainingSessions = sessionList.filter { !sessionsToDelete.contains(it) }
                val sortedByAgeAndSize = remainingSessions.sortedWith(
                    compareBy<SessionInfo> { it.dateTime }.thenByDescending { it.sizeBytes },
                )

                val maxAdditionalSessions = (sessionList.size * 0.3).toInt()
                sessionsToDelete.addAll(sortedByAgeAndSize.take(maxAdditionalSessions))
            }

            if (sessionsToDelete.isEmpty()) {
                Toast.makeText(requireContext(), "No sessions selected for cleanup", Toast.LENGTH_SHORT).show()
                return
            }

            val totalSizeToDeleteMB = sessionsToDelete.sumOf { it.sizeBytes } / (1024 * 1024)

            val confirmMessage = buildString {
                append("Cleanup ${sessionsToDelete.size} sessions?\n\n")
                append("• Age-based: ${oldSessions.size} sessions older than $defaultCleanupDays days\n")
                if (shouldCleanupBySize) {
                    append("• Storage-based: Additional sessions (storage $usagePercent% full)\n")
                }
                append("• Total space to free: ${totalSizeToDeleteMB}MB\n")
                append("• Remaining sessions: ${sessionList.size - sessionsToDelete.size}")
            }

            android.app.AlertDialog.Builder(requireContext())
                .setTitle("Cleanup Old Sessions")
                .setMessage(confirmMessage)
                .setPositiveButton("Delete") { _, _ ->
                    performCleanup(sessionsToDelete)
                }
                .setNegativeButton("Cancel", null)
                .show()
        } catch (e: Exception) {
            android.util.Log.e("FileManager", "Error in cleanup analysis: ${e.message}", e)
            Toast.makeText(requireContext(), "Error analyzing sessions for cleanup: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun performCleanup(sessionsToDelete: List<SessionInfo>) {
        try {
            var deletedCount = 0
            var deletedSizeMB = 0L
            var errorCount = 0

            sessionsToDelete.forEach { session ->
                try {
                    val sizeMB = session.sizeBytes / (1024 * 1024)
                    if (session.directory.deleteRecursively()) {
                        deletedCount++
                        deletedSizeMB += sizeMB
                        android.util.Log.d("FileManager", "Deleted session: ${session.name} (${sizeMB}MB)")
                    } else {
                        errorCount++
                        android.util.Log.w("FileManager", "Failed to delete session: ${session.name}")
                    }
                } catch (e: Exception) {
                    errorCount++
                    android.util.Log.e("FileManager", "Error deleting session ${session.name}: ${e.message}", e)
                }
            }

            val resultMessage = buildString {
                append("Cleanup completed:\n")
                append("• Deleted: $deletedCount sessions (${deletedSizeMB}MB)\n")
                if (errorCount > 0) {
                    append("• Errors: $errorCount sessions\n")
                }
                append("• Storage freed: ${deletedSizeMB}MB")
            }

            Toast.makeText(requireContext(), resultMessage, Toast.LENGTH_LONG).show()

            loadSessions()
        } catch (e: Exception) {
            android.util.Log.e("FileManager", "Error during cleanup: ${e.message}", e)
            Toast.makeText(requireContext(), "Cleanup failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        fun newInstance(): FileManagerFragment = FileManagerFragment()
    }
}
