# Live camera

Set `LIVE_CAMERA_URL` in the application's private `.env` to the configured
camera's RTSP address. The server must be able to reach it. From the codebase
directory, run the live API alongside the existing Streamlit app, video API
and video worker:

    python -m uvicorn livestreaming.main:app --host 127.0.0.1 --port 8011

Use one live API process. Both backends must run on the same Linux host and
use the same local `DATA_DIR` so their processing locks work. Restart the
updated video worker before using live mode. The frontend uses
`LIVE_BACKEND_URL` (default http://127.0.0.1:8011).

The repository environment pins CPU PyTorch. On the GPU host, install matching
CUDA-enabled PyTorch and torchvision builds compatible with the GPU driver in
the live service's environment. Run this with that environment's Python:

    python -c "import torch; print(torch.cuda.is_available())"

Check that it prints True and set `DEVICE=cuda` for the live API process.
Verify camera-to-screen delay on that host. The reader tracks accumulated
processing delay relative to the stream clock, which excludes any delay
already present when the first camera picture arrives.

Choose **Live camera**, get the preview, mark four corners, set the people
count and duration, and press **Start monitoring**. Enable sound in the
viewer to hear the current alarm. Refreshing the page reconnects to the
session. An interrupted camera connection starts fresh tracking on reconnect.
If picture dimensions change, get a fresh preview and mark the zone again.

Stop ends the session and releases waiting video jobs. Completed alarms,
including the last annotated JPEG and end reason, are stored in
`DATA_DIR/live-events.sqlite3`. Load recent alarm events in the live view
to inspect them. Stream seconds count from the start of each continuous
connection. The UTC time in the event list is the backend's alarm trigger time.
An abrupt process exit can lose the currently open alarm. Restart monitoring
after a live API restart.
