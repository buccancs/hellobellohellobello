package com.yourcompany.sensorspoke.ui.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.yourcompany.sensorspoke.R
import com.yourcompany.sensorspoke.ui.models.SessionInfo
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * RecyclerView adapter for displaying session information
 */
class SessionAdapter(
    private val sessions: List<SessionInfo>,
    private val onSessionClick: (SessionInfo) -> Unit,
) : RecyclerView.Adapter<SessionAdapter.SessionViewHolder>() {
    private val dateFormat = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())

    class SessionViewHolder(
        itemView: View,
    ) : RecyclerView.ViewHolder(itemView) {
        val sessionNameText: TextView = itemView.findViewById(R.id.sessionNameText)
        val sessionSizeText: TextView = itemView.findViewById(R.id.sessionSizeText)
        val sessionDateText: TextView = itemView.findViewById(R.id.sessionDateText)
        val sessionDetailsText: TextView = itemView.findViewById(R.id.sessionDetailsText)
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int,
    ): SessionViewHolder {
        val view =
            LayoutInflater
                .from(parent.context)
                .inflate(R.layout.item_session, parent, false)
        return SessionViewHolder(view)
    }

    override fun onBindViewHolder(
        holder: SessionViewHolder,
        position: Int,
    ) {
        val session = sessions[position]

        holder.sessionNameText.text = session.name
        holder.sessionSizeText.text = formatSize(session.sizeBytes)
        holder.sessionDateText.text = dateFormat.format(session.dateTime)
        holder.sessionDetailsText.text = session.details

        holder.itemView.setOnClickListener {
            onSessionClick(session)
        }
    }

    override fun getItemCount(): Int = sessions.size

    private fun formatSize(bytes: Long): String =
        when {
            bytes < 1024 -> "$bytes B"
            bytes < 1024 * 1024 -> "${bytes / 1024} KB"
            bytes < 1024 * 1024 * 1024 -> "${bytes / (1024 * 1024)} MB"
            else -> "${"%.1f".format(bytes / (1024.0 * 1024.0 * 1024.0))} GB"
        }
}
