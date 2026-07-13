import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch


PREDICTION_KEYS = ("target_bbox", "best_score", "all_boxes", "all_scores")
PREDICTION_KEY_SET = frozenset(PREDICTION_KEYS)
ROOT_KEYS = frozenset(("frame_index", "previous_output"))


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _freeze_vector(value, name):
    if torch.is_tensor(value):
        if value.requires_grad:
            raise ValueError(f"{name} tensor must not require gradients")
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} tensor must be real numeric")
        if value.ndim != 1:
            raise ValueError(f"{name} tensor must be one-dimensional")
        value = value.detach().cpu().tolist()
    elif not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a tensor or numeric sequence")
    return tuple(_finite_number(item, name) for item in value)


def _freeze_prediction_value(key, value):
    if key == "best_score":
        return _finite_number(value, key)
    frozen = _freeze_vector(value, key)
    if key == "target_bbox" and len(frozen) != 4:
        raise ValueError("target_bbox must contain exactly one xywh box")
    if key == "all_boxes" and (not frozen or len(frozen) % 4 != 0):
        raise ValueError("all_boxes must contain one or more flat xywh boxes")
    if key == "all_scores" and not frozen:
        raise ValueError("all_scores must be a non-empty flat vector")
    return frozen


def sanitize_previous_output(previous_output):
    if not isinstance(previous_output, Mapping):
        raise TypeError("previous_output must be a mapping")
    safe = {}
    for key in PREDICTION_KEYS:
        if key in previous_output:
            safe[key] = _freeze_prediction_value(key, previous_output[key])
    return safe


def freeze_prediction_history(previous_output, required_keys=("target_bbox",)):
    if not isinstance(required_keys, (tuple, list, frozenset)):
        raise TypeError("required_keys must be a finite key collection")
    required = frozenset(required_keys)
    if not required.issubset(PREDICTION_KEY_SET):
        raise ValueError("required_keys contains an unsupported key")
    safe = sanitize_previous_output(previous_output)
    missing = required.difference(safe)
    if missing:
        raise ValueError(f"missing required prediction keys: {sorted(missing)}")
    return MappingProxyType(safe)


def sanitize_initialization_info(info):
    if not isinstance(info, Mapping):
        raise TypeError("initialization info must be a mapping")
    if set(info) != {"init_bbox"}:
        raise ValueError("initialization accepts only init_bbox")
    box = info["init_bbox"]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise TypeError("init_bbox must be one xywh sequence")
    box = [_finite_number(item, "init_bbox") for item in box]
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError("init_bbox width and height must be positive")
    return {"init_bbox": box}


def assert_causal_tracker_info(info):
    if not isinstance(info, Mapping):
        raise TypeError("tracker info must be a mapping")
    if set(info) != ROOT_KEYS:
        raise ValueError("unexpected tracker-info keys")
    previous = info["previous_output"]
    if not isinstance(previous, Mapping):
        raise TypeError("previous_output must be a mapping")
    if not set(previous).issubset(PREDICTION_KEY_SET):
        raise ValueError("unexpected prediction-history keys")
    CausalFrameRecord.from_evaluator(info["frame_index"], previous)


@dataclass(frozen=True)
class CausalFrameRecord:
    frame_index: int
    previous_output: Mapping

    def __post_init__(self):
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an int")
        if self.frame_index < 1:
            raise ValueError("frame_index must be positive")
        frozen = freeze_prediction_history(self.previous_output)
        object.__setattr__(self, "previous_output", frozen)

    @classmethod
    def from_evaluator(cls, frame_index, previous_output):
        return cls(frame_index, previous_output)

    def as_tracker_info(self):
        return {
            "frame_index": self.frame_index,
            "previous_output": dict(self.previous_output),
        }


def call_causal_track(tracker, image, record):
    if not isinstance(record, CausalFrameRecord):
        raise TypeError("record must be CausalFrameRecord")
    safe_record = CausalFrameRecord(record.frame_index, record.previous_output)
    return tracker.track(image, info=safe_record.as_tracker_info())
