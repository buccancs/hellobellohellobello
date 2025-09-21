# Data Backup Strategy (3-2-1 Rule)

This document outlines the recommended backup strategy for the Multi-Modal Physiological Sensing Platform. The data
collected is critical for research and must be safeguarded.

## The 3-2-1 Backup Rule

- Keep 3 total copies of your data (1 primary + 2 backups).
- Store the copies on at least 2 different types of media (e.g., internal SSD + external HDD/NAS).
- Keep at least 1 copy off-site (e.g., cloud storage or a drive stored in a different location).

## How to Use This Template

Fill in the sections below to document your lab/research group's specific backup plan.

### 1. Primary Data Location

- Primary Session Data Folder: <path to your session data folder>
- Machine Name / Host: <hostname of PC Controller>

### 2. Local Backup (Secondary Copy)

- Destination Type: <External HDD/NAS>
- Device/Share Name: <e.g., WD_Elements_8TB>
- Mount/Drive Letter: <e.g., E:\>
- Backup Destination Folder: <e.g., E:\ResearchBackups\PhysioPlatform>
- Backup Schedule: <e.g., Daily at 2:00 AM via Task Scheduler>

### 3. Off-Site Backup (Tertiary Copy)

- Provider or Storage Type: <e.g., Cloud Provider or Off-site HDD>
- Encryption: <e.g., Provider-side encryption + local ZIP AES256>
- Rotation Policy: <e.g., weekly full backup, monthly archive>

### 4. Verification and Restore Testing

- Integrity Check Method: <e.g., checksums, comparing file counts>
- Restore Test Frequency: <e.g., quarterly dry-run restore>

### 5. Retention Policy

- Retention Duration: <e.g., keep last 6 months of daily backups and last 2 years of monthly backups>

## Automation Script

A helper script is provided at `scripts/backup_script.py` to automate copying the session data to a local backup
destination.

### Usage (Windows PowerShell example)

```
python scripts\backup_script.py --source "C:\\Data\\PhysioSessions" --dest "E:\\ResearchBackups\\PhysioPlatform" --log "C:\\Data\\backup_logs\\backup.log"
```

### Scheduling on Windows (Task Scheduler)

1. Open Task Scheduler > Create Basic Task.
2. Trigger: Daily (or as needed).
3. Action: Start a Program, set Program/script to `python` and Add arguments to the command above.
4. Start in: project root directory.

### Scheduling on macOS/Linux (cron)

```
crontab -e
0 2 * * * /usr/bin/python3 /path/to/repo/scripts/backup_script.py --source "/data/PhysioSessions" --dest "/mnt/backup/PhysioPlatform" --log "/data/backup_logs/backup.log" >> /var/log/physio_backup_cron.log 2>&1
```
