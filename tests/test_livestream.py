import json
from pathlib import Path
import threading
import time
from fractions import Fraction
from unittest.mock import Mock

import av
import numpy as np
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.services.cv.annotation import render
from backend.services.cv.live_pipeline import process
from backend.services.locks import ProcessLock
from backend.workers.worker import Worker

ZONE = [[1, 1], [119, 1], [119, 119], [1, 119]]
BODY = dict(polygon=ZONE, n_people=1, threshold_seconds=.2, width=120, height=120)


class Detector:
    image_size = 640
    weights = 'test'
    device = 'cpu'

    def __init__(self):
        self.i = 0

    def predict(self, image):
        inside = self.i < 6 or self.i >= 16
        self.i += 1
        return [dict(xyxy=[20, 20, 70, 90], score=.95)] if inside else []


def test_live_matches_file_counts_states_and_alarm_edges(tmp_path):
    from livestreaming.pipeline import LiveFrames
    path = tmp_path / 'fixture.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width = stream.height = 120
        for i in range(24):
            image = np.full((120, 120, 3), 40, np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format='bgr24')))
        container.mux(stream.encode())
    settings = Settings(data_dir=tmp_path)
    report = process(path, ZONE, 1, .2, Detector(), tmp_path, write_video=False,
                     evidence_mode=True, stable_ids=True, zone_policy=settings.zone_policy)
    records = [json.loads(line) for line in (Path(report['output_dir'])/'diagnostics.jsonl').read_text().splitlines()]
    live = LiveFrames(settings, BODY, 10, Detector())
    alarm_active = False
    with av.open(str(path)) as container:
        for frame, expected in zip(container.decode(video=0), records, strict=True):
            image = frame.to_ndarray(format='bgr24')
            annotated, result = live.step(image, frame.pts * frame.time_base)
            alarm_active = bool(expected['count'] and (alarm_active or expected['alarm']))
            reference = render(image, ZONE, expected['tracks'], 1, expected['observed_count'],
                               expected['count'], expected['elapsed_seconds'], .2,
                               expected['video_seconds'], alarm_active, overlap=settings.overlap_fraction)
            np.testing.assert_array_equal(annotated, reference)
            for key in ('count', 'observed_count', 'elapsed_seconds', 'alarm'):
                assert result[key] == expected[key]
    assert len(report['events']) == 2


def test_stream_api_rearms_saves_last_active_frame_and_stops(tmp_path):
    from livestreaming.main import create_app
    reached_end = threading.Event()
    release = threading.Event()

    def frames():
        for i in range(24):
            yield np.full((120,120,3), 40, np.uint8), Fraction(i,10), 10
        reached_end.set()
        release.wait(5)

    with TestClient(create_app(Settings(data_dir=tmp_path), frames, Detector)) as api:
        assert api.get('/preview').json()['width'] == 120
        assert api.post('/start', json=BODY).status_code == 202
        assert reached_end.wait(5)
        assert api.post('/start', json=BODY).status_code == 409
        status = api.get('/status').json()
        assert status['state'] == 'running' and status['frame'] and status['active_alarm']
        api.post('/stop')
        release.set()
        for _ in range(100):
            if api.get('/status').json()['state'] == 'stopped':
                break
            time.sleep(.02)
        events = api.get('/events').json()
        assert len(events) == 2
        assert events[0]['reason'] == 'stopped'
        assert events[1]['reason'] == 'count_below_threshold'
        assert events[1]['screenshot_seconds'] < events[1]['end_seconds']
        assert api.get(events[1]['image_url']).content.startswith(b'\xff\xd8')


def test_video_waits_for_live_reservation(tmp_path):
    repo = Mock()
    worker = Worker(Settings(data_dir=tmp_path), repository=repo, pipeline=Mock(), storage=Mock())
    with ProcessLock(tmp_path/'live.lock'):
        assert not worker.run_once()
    repo.claim_job.assert_not_called()


def test_frontend_offers_live_mode():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file('../frontend/app.py').run()
    assert 'Live camera' in app.radio[0].options


def test_timestamp_gap_ends_alarm_and_reconnect_starts_at_zero(tmp_path):
    from livestreaming.main import create_app
    connected_twice = threading.Event()
    release = threading.Event()
    attempts = 0

    def frames():
        nonlocal attempts
        attempts += 1
        stamps = [0, 1, 2, 3, 19] if attempts == 1 else [0, 1]
        for tick in stamps:
            yield np.zeros((120, 120, 3), np.uint8), Fraction(tick, 10), 10
        if attempts > 1:
            connected_twice.set()
            release.wait(5)

    with TestClient(create_app(Settings(data_dir=tmp_path), frames, Detector)) as api:
        api.post('/start', json=BODY).raise_for_status()
        assert connected_twice.wait(5)
        result = api.get('/status').json()
        assert result['seconds'] == .1 and not result['active_alarm']
        event, = api.get('/events').json()
        assert event['reason'] == 'interrupted'
        assert event['screenshot_seconds'] == .3
        api.post('/stop')
        release.set()


def test_live_waits_for_current_video_and_can_cancel_without_camera(tmp_path):
    from livestreaming.main import create_app
    frames = Mock()
    with TestClient(create_app(Settings(data_dir=tmp_path), frames, Detector)) as api:
        with ProcessLock(tmp_path/'processing.lock'):
            api.post('/start', json=BODY).raise_for_status()
            assert api.get('/status').json()['state'] == 'waiting'
            api.post('/stop')
            for _ in range(100):
                if api.get('/status').json()['state'] == 'stopped':
                    break
                time.sleep(.01)
            assert api.get('/status').json()['state'] == 'stopped'
    frames.assert_not_called()
