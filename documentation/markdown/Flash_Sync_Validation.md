# Flash Sync Validation Procedure

This document describes how to validate end-to-end temporal synchronization across all sensor streams using the Flash
Sync event.

## Objective

Verify that timestamps across RGB video, thermal frames, and GSR data align within ±5 ms (FR7), using the synchronized
on-screen flash.

## Prerequisites

- PC Controller (Hub) running on the same network as Android device(s).
- Android Sensor Node app installed and connected.
- Sufficient ambient lighting; point the RGB/thermal cameras at the Android device screen.

## Steps

1. Start the PC Controller and connect at least one Android Spoke.
2. Initiate a session and begin recording (Start Recording).
3. Trigger the Flash Sync from the Hub (Flash Sync button). All Spokes display a full-screen white flash.
4. Trigger multiple flashes (e.g., 5 times) spaced a few seconds apart for better statistics.
5. Stop recording; allow the Spokes to transfer files to the Hub.

## Analysis

1. Locate the following files in the session directory on the Hub:
    - RGB video (MP4) per device
    - Thermal CSV per device
    - GSR CSV per device
    - flash_sync_events.csv from each device
2. For each flash event time (from flash_sync_events.csv), measure the observed flash onset time in the RGB video (frame
   timestamp where the brightness spikes) and in the thermal data (frame where average pixel temperature/brightness
   spikes).
3. Compute the PC-vs-device clock offset using the NTP-like handshake logs stored during connection (NetworkController).
   Apply the offset to align device timestamps to the PC reference timeline.
4. Calculate the absolute difference between the aligned timestamps for RGB, thermal, and GSR signals at each flash
   event.
5. Confirm that the mean and 95th percentile of the differences are within ±5 ms.

## Tips

- Use a short exposure in RGB to reduce motion blur in the flash onset.
- Ensure the phone screen is clearly visible in the camera FOV.
- Repeat the test after any network changes or time-sync configuration updates.

## Automation (Optional)

Create a small Python script to:

- Parse flash_sync_events.csv,
- Compute brightness delta across video frames (OpenCV),
- Detect the first frame above a threshold around each event,
- Compare timestamps with applied clock offset,
- Emit a report with mean/median/percentiles and pass/fail verdict.
