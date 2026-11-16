"""One configured camera, one monitoring session, one API process."""
import logging
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack, asynccontextmanager, closing
from datetime import datetime, timezone

import av
import cv2
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, FiniteFloat, SecretStr

from backend.core.config import Settings, VIDEO_FALLBACK_FPS
from backend.services.cv.media_utils import validate_polygon
from backend.services.locks import LockBusy, ProcessLock
from backend.services.media import image_data_uri
from livestreaming.pipeline import LiveFrames, prepare_detector

log = logging.getLogger(__name__)


class LiveSettings(Settings):
    live_camera_url: SecretStr = SecretStr('')
    live_delay_limit_seconds: float = Field(default=2, gt=0)


class Rule(BaseModel):
    polygon: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=4, max_length=4)
    n_people: int = Field(strict=True, gt=0)
    threshold_seconds: FiniteFloat = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CameraInterrupted(Exception):
    pass


class ViewChanged(Exception):
    pass


def decoded_camera_frames(settings):
    url = settings.live_camera_url.get_secret_value()
    if not url:
        raise CameraInterrupted('Configure LIVE_CAMERA_URL on the live backend.')
    try:
        with av.open(url, options={'rtsp_transport': 'tcp'}, timeout=(5, 5)) as source:
            stream = source.streams.video[0]
            fps = float(stream.average_rate or VIDEO_FALLBACK_FPS)
            # Joining an HEVC stream can precede its first usable reference
            # picture. Keep the decoder connected long enough to recover.
            deadline = time.monotonic() + 5
            started = False
            for packet in source.demux(stream):
                if not started and time.monotonic() >= deadline:
                    raise CameraInterrupted('Camera startup timed out waiting for a usable frame.')
                try:
                    frames = packet.decode()
                except av.InvalidDataError:
                    if started:
                        raise CameraInterrupted('Camera decoding interrupted. Restarting tracking.') from None
                    continue
                for frame in frames:
                    if frame.pts is None or frame.is_corrupt:
                        if started:
                            raise CameraInterrupted('Camera returned an unusable frame. Restarting tracking.')
                        continue
                    started = True
                    yield frame.to_ndarray(format='bgr24'), frame.pts * frame.time_base, fps
            if not started:
                raise CameraInterrupted('Camera startup ended without a usable frame.')
    except (av.FFmpegError, OSError, IndexError):
        raise CameraInterrupted('Camera connection interrupted.') from None


def camera_frames(settings):
    """Drain camera decoding independently; inference always takes the newest frame."""
    pending = queue.Queue(maxsize=1)
    stopped = threading.Event()

    def publish(item):
        while not stopped.is_set():
            try:
                pending.put_nowait(item)
                return
            except queue.Full:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    pass

    def read():
        try:
            with closing(decoded_camera_frames(settings)) as source:
                for frame in source:
                    if stopped.is_set():
                        break
                    publish(frame)
        except Exception as exc:
            publish(exc)
        else:
            publish(CameraInterrupted('Camera stream ended. Reconnecting.'))

    reader = threading.Thread(target=read, name='camera-reader', daemon=True)
    reader.start()
    try:
        while True:
            try:
                item = pending.get(timeout=12)
            except queue.Empty:
                raise CameraInterrupted('Camera reader timed out.') from None
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stopped.set()
        reader.join(timeout=6)


def jpeg(image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('Could not encode camera image.')
    return encoded.tobytes()


class Monitor:
    def __init__(self, settings, frames, detector_factory):
        self.settings = settings
        self.frames = frames
        self.detector_factory = detector_factory
        self.mutex = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.status = dict(state='stopped', active_alarm=False, frame=None)
        self.database = settings.data_dir / 'live-events.sqlite3'

    def update(self, **values):
        with self.mutex:
            self.status.update(values)

    def start(self, body):
        with self.mutex:
            if self.thread and self.thread.is_alive():
                raise HTTPException(409, 'A live session already exists.')
            self.cancel.clear()
            self.status = dict(state='waiting', frame=None, active_alarm=False, message='Waiting for processing capacity.')
            self.thread = threading.Thread(target=self.run, args=(body,), name='live-camera')
            self.thread.start()

    def stop(self):
        with self.mutex:
            self.cancel.set()
            if self.thread and self.thread.is_alive():
                self.status.update(state='stopping', active_alarm=False)

    def reserve(self, locks, filename):
        while not self.cancel.is_set():
            try:
                locks.enter_context(ProcessLock(self.settings.data_dir / filename))
                return True
            except LockBusy:
                self.cancel.wait(.2)
        return False

    def run(self, body):
        terminal, message = 'stopped', 'Monitoring stopped.'
        try:
            with ExitStack() as locks:
                if not all(self.reserve(locks, name) for name in ('live.lock', 'processing.lock')):
                    return
                self.update(state='starting', message='Preparing detector.')
                detector = self.detector_factory() if self.detector_factory else prepare_detector(self.settings, body)
                self.update(device=getattr(detector, 'device', 'unknown'))
                log.info('Live detector ready on %s', getattr(detector, 'device', 'unknown'))
                session = uuid.uuid4().hex
                while not self.cancel.is_set():
                    try:
                        self.watch(body, detector, session)
                    except CameraInterrupted as exc:
                        self.update(state='reconnecting', frame=None, active_alarm=False, message=str(exc))
                        self.cancel.wait(1)
        except ViewChanged:
            terminal, message = 'error', 'Camera dimensions changed. Get a new preview and draw the zone again.'
        except Exception:
            log.exception('Live processing failed')
            terminal, message = 'error', 'Live processing failed. Check the backend log.'
        finally:
            self.update(state=terminal, message=message, frame=None, active_alarm=False)

    def watch(self, body, detector, session):
        origin = previous = engine = None
        started = time.monotonic()
        last_active = None
        try:
            with closing(self.frames()) as frames:
                for image, stamp, fps in frames:
                    if self.cancel.is_set():
                        break
                    if image.shape[:2] != (body['height'], body['width']):
                        raise ViewChanged()
                    if origin is None:
                        origin, started = stamp, time.monotonic()
                        engine = LiveFrames(self.settings, body, fps, detector)
                    elif stamp <= previous or stamp - previous > self.settings.max_frame_gap_seconds:
                        raise CameraInterrupted('Camera timestamp gap. Restarting tracking.')
                    previous = stamp
                    t = stamp - origin
                    annotated, result = engine.step(image, t)
                    picture = jpeg(annotated)
                    if time.monotonic() - started - float(t) > getattr(self.settings, 'live_delay_limit_seconds', 2):
                        raise CameraInterrupted('Processing fell behind. Reconnecting.')
                    if result['alarm']:
                        alarm_start = float(t)
                        alarm_time = datetime.now(timezone.utc).isoformat()
                    if result['active_alarm']:
                        last_active = float(t), picture
                    elif last_active is not None:
                        self.save_event(session, alarm_time, alarm_start, float(t), last_active, 'count_below_threshold')
                        last_active = None
                    self.update(**result, frame=image_data_uri(picture), state='running',
                                updated=time.time(), message='Monitoring.',
                                processing_delay_seconds=max(0, time.monotonic() - started - float(t)))
                if not self.cancel.is_set():
                    raise CameraInterrupted('Camera stream ended. Reconnecting.')
        finally:
            if last_active is not None:
                reason = 'stopped' if self.cancel.is_set() else 'interrupted'
                self.save_event(session, alarm_time, alarm_start, last_active[0], last_active, reason)

    def save_event(self, session, utc, start, end, last_active, reason):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute('INSERT INTO events(session,utc,start_seconds,end_seconds,screenshot_seconds,reason,picture) '
                       'VALUES(?,?,?,?,?,?,?)', (session, utc, start, end, last_active[0], reason, last_active[1]))


def create_app(settings=None, frames=None, detector_factory=None):
    settings = settings or LiveSettings()
    monitor = Monitor(settings, frames or (lambda: camera_frames(settings)), detector_factory)

    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with ProcessLock(settings.data_dir / 'live-api.lock'):
            with closing(sqlite3.connect(monitor.database)) as db, db:
                db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session TEXT, utc TEXT, '
                           'start_seconds REAL, end_seconds REAL, screenshot_seconds REAL, reason TEXT, picture BLOB)')
            try:
                yield
            finally:
                monitor.stop()
                if monitor.thread:
                    monitor.thread.join()

    app = FastAPI(lifespan=lifespan, title='Live camera')

    @app.get('/preview')
    def preview():
        try:
            with closing(monitor.frames()) as source:
                image, _, _ = next(source)
        except (CameraInterrupted, StopIteration) as exc:
            raise HTTPException(503, str(exc) or 'No camera picture available.') from None
        h, w = image.shape[:2]
        return dict(width=w, height=h, first_frame=image_data_uri(jpeg(image)), upload_id=uuid.uuid4().hex)

    @app.post('/start', status_code=202)
    def start(body: Rule):
        try:
            validate_polygon(body.polygon, body.width, body.height)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if frames is None and not settings.live_camera_url.get_secret_value():
            raise HTTPException(503, 'Configure LIVE_CAMERA_URL on the live backend.')
        monitor.start(body.model_dump())
        return {'accepted': True}

    @app.post('/stop')
    def stop():
        monitor.stop()
        return {'accepted': True}

    @app.get('/status')
    def status():
        with monitor.mutex:
            return monitor.status.copy()

    @app.get('/events')
    def events():
        with closing(sqlite3.connect(monitor.database)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('SELECT id,session,utc,start_seconds,end_seconds,screenshot_seconds,reason '
                              'FROM events ORDER BY id DESC LIMIT 50')
            return [dict(row, image_url=f"/events/{row['id']}/image") for row in rows]

    @app.get('/events/{event_id}/image')
    def screenshot(event_id: int):
        with closing(sqlite3.connect(monitor.database)) as db:
            row = db.execute('SELECT picture FROM events WHERE id=?', (event_id,)).fetchone()
        if row is None:
            raise HTTPException(404, 'Event not found.')
        return Response(row[0], media_type='image/jpeg')

    return app


app = create_app()
