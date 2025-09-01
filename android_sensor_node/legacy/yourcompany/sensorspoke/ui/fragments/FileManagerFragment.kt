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

            // Sort by date (newest first)
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

        // Count files
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
        // Extract datetime from session name format: session_YYYYMMDD_HHMMSS
        val pattern = """session_(\d{8})_(\d{6})""".toRegex()
        val matchResult = pattern.find(sessionName)

        return if (matchResult != null) {
            val (dateStr, timeStr) = matchResult.destructured
            val dateTimeStr = "${dateStr}_$timeStr"
            val format = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            format.parse(dateTimeStr) ?: Date()
        } else {
            Date() // Fallback to current time
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
        // Show session details or navigate to details view
        Toast
            .makeText(
                requireContext(),
                "Session: ${session.name}\nSize: ${session.sizeBytes / (1024 * 1024)} MB",
                Toast.LENGTH_LONG,
            ).show()
    }

    private fun cleanupOldSessions() {
        try {
            val cutoffDate =
                Calendar
                    .getInstance()
                    .apply {
                        add(Calendar.DAY_OF_MONTH, -7) // Keep sessions from last 7 days
                    }.time

            val oldSessions = sessionList.filter { it.dateTime.before(cutoffDate) }

            if (oldSessions.isEmpty()) {
                Toast.makeText(requireContext(), "No old sessions to cleanup", Toast.LENGTH_SHORT).show()
                return
            }

            var deletedCount = 0
            oldSessions.forEach { session ->
                try {
                    if (session.directory.deleteRecursively()) {
                        deletedCount++
                    }
                } catch (e: Exception) {
                    // Continue with other sessions
                }
            }

            Toast
                .makeText(
                    requireContext(),
                    "Deleted $deletedCount old sessions",
                    Toast.LENGTH_SHORT,
                ).show()

            loadSessions() // Refresh the list
        } catch (e: Exception) {
            Toast.makeText(requireContext(), "Cleanup error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        fun newInstance(): FileManagerFragment = FileManagerFragment()
    }
}
