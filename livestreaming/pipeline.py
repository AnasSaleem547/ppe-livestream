"""Apply the video backend's detection and zone rules to one live frame."""
import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from backend.core import config as cfg
from backend.services.cv.annotation import render
from backend.services.cv.cover_evidence import CoverEvidence
from backend.services.cv.identity_link import IdentityLinker
from backend.services.cv.live_pipeline import Episode, filter_boxes, make_tracker
from backend.services.cv.media_utils import padded_crop
from backend.services.cv.occupancy import Occupancy
from backend.services.cv.visual_support import VisualSupport
from backend.services.pipeline import PipelineRunner


def crop_for(settings, body):
    width, height = body['width'], body['height']
    if settings.full_frame_context:
        return [0, 0, width, height]
    return padded_crop(body['polygon'], width, height)


def prepare_detector(settings, body):
    runner = PipelineRunner(settings)
    runner._prepare_detector(crop_for(settings, body), body['width'], body['height'])
    if settings.device == 'cuda' and runner.detector.device != 'cuda':
        raise RuntimeError('GPU warmup failed.')
    return runner.detector


class LiveFrames:
    def __init__(self, settings, body, fps, detector):
        self.detector = detector
        self.zone = body['polygon']
        self.n = body['n_people']
        self.target = body['threshold_seconds']
        self.crop = crop_for(settings, body)
        self.tracker = make_tracker(fps, self.crop)
        self.identity = IdentityLinker()
        self.identity.bounds = self.crop
        self.cover = CoverEvidence()
        self.visual = VisualSupport(settings.max_image_seconds)
        self.occupancy = Occupancy(
            self.zone, settings.overlap_fraction, settings.max_hidden_seconds,
            evidence_mode=cfg.PIPELINE_EVIDENCE_MODE, max_extent_seconds=settings.max_body_reference_seconds,
            membership_mode=settings.zone_membership_mode, anchor=settings.zone_anchor,
            boundary_margin_fraction=settings.zone_boundary_margin_fraction,
            boundary_max_seconds=settings.zone_boundary_max_seconds,
        )
        self.episode = Episode(self.n, self.target)

    def step(self, image, stamp):
        seconds = float(stamp)
        left, top, right, bottom = self.crop
        raw = self.detector.predict(image[top:bottom, left:right])
        boxes = filter_boxes([dict(d, xyxy=(np.asarray(d['xyxy']) + [left, top, left, top]).tolist())
                              for d in raw])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cfg.PIPELINE_EVIDENCE_MODE:
            self.cover.update(image)
        rows = np.asarray([[*d['xyxy'], d['score'], 0] for d in boxes], np.float32).reshape(-1, 6)
        tracked = self.tracker.update(Boxes(rows, image.shape[:2]))
        observed = [dict(xyxy=row[:4].tolist(), id=int(row[4]), score=float(row[5])) for row in tracked]
        if cfg.PIPELINE_STABLE_IDS:
            observed = self.identity.update(seconds, observed, image, protected=set(self.occupancy.people))
        tracks = self.occupancy.step(
            seconds, observed, occluders=boxes,
            verify_visible=lambda person: self.visual.verify(person, gray, seconds, occupied=observed),
            cover_check=self.cover.covers if cfg.PIPELINE_EVIDENCE_MODE else None,
        )
        self.cover.prune(set(self.occupancy.people))
        if cfg.PIPELINE_STABLE_IDS:
            self.identity.remember(seconds, tracks, image)
        count = sum(person['inside'] for person in tracks)
        confirmed = sum(person['inside'] and person['state'] == 'visible_inside' for person in tracks)
        elapsed, alarm = self.episode.step(stamp, count, confirmed)
        for person in observed:
            self.visual.remember(person, gray, seconds, update_template=person['score'] >= cfg.VISUAL_TEMPLATE_MIN_CONFIDENCE)
        active = self.episode.gate.fired
        picture = render(image, self.zone, tracks, self.n, confirmed, count, float(elapsed),
                         self.target, seconds, active, overlap=self.occupancy.overlap)
        return picture, dict(count=count, observed_count=confirmed, elapsed_seconds=float(elapsed),
                             alarm=bool(alarm), active_alarm=bool(active), seconds=seconds)
