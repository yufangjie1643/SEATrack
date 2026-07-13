# Target-Spectral A0 Causal and Config Prerequisites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ground-truth/lifecycle leakage from every in-repository SEATrack evaluation entry point and make evaluation configuration loads deeply isolated, without changing network forward arithmetic or weights.

**Architecture:** A frozen causal record whitelists only frame index and copied, finite, shape-checked prediction history; tensors carrying gradients are rejected. OPE, video, and VOT share one keyword-only call helper and an explicit episode reset contract. Evaluation YAMLs are merged into a clone of a pristine import-time default snapshot rather than the mutable module singleton.

**Tech Stack:** Python 3.12, PyTorch, EasyDict/YAML, `unittest`, `unittest.mock`.

## Global Constraints

- Work only in `.worktrees/target-spectral-a` on branch `target-spectral-a`.
- Use `/home/yufan/code/SEATrack/.venv/bin/python`; the worktree has no private virtualenv.
- Do not modify model, HMoE, checkpoint, padding-mask, CE, training, or target-spectral routing code in this child plan.
- Tracker-side code may receive the legal initialization box only in `initialize`; after initialization it must never read GT, visibility, validity, attributes, corruption metadata, or future data.
- This child plan supports single-object, frame-zero initialization. Other protocols fail closed rather than silently changing semantics.
- Existing routing-disabled SEATrack prediction arithmetic must remain textually unchanged inside `track`; only argument validation, frame bookkeeping, and entry-point calls change.
- Every security validation uses explicit `TypeError`, `ValueError`, or `RuntimeError`; never Python `assert`.
- Each task follows RED, minimal GREEN, focused regression, full regression, review, then task-local commit.

---

### Task 1: Causal Schema and Real Episode Entry Points

**Files:**

- Create: `lib/test/evaluation/causal.py`
- Create: `tests/test_spectral_causality.py`
- Modify: `lib/test/evaluation/tracker.py`
- Modify: `lib/test/tracker/basetracker.py`
- Modify: `lib/test/tracker/seatrack.py`
- Modify: `lib/test/tracker/ostrack.py`
- Modify: `lib/test/vot/seatrack_class.py`

**Interfaces:**

- `CausalFrameRecord.from_evaluator(frame_index, previous_output)` returns an immutable sanitized record.
- `CausalFrameRecord.as_tracker_info()` returns a fresh mapping containing exactly `frame_index` and `previous_output`.
- `freeze_prediction_history(previous_output, required_keys=("target_bbox",))` immediately validates and deep-freezes one successful tracker output for atomic evaluator-side commit; VOT additionally requires `best_score`.
- `assert_causal_tracker_info(info)` raises on any schema/value violation.
- `sanitize_initialization_info(info)` accepts only one finite positive-size `init_bbox` and returns a fresh list.
- `call_causal_track(tracker, image, record)` calls `tracker.track(image, info=...)` by keyword.
- `BaseTracker.begin_episode(reset_global=True)` marks exactly one initialization pending.
- `BaseTracker._prepare_episode_initialization()` performs the direct-call fallback and consumes that pending initialization exactly once.
- `BaseTracker._validate_causal_frame(info)` rejects uninitialized/out-of-order calls before image/crop access without mutating state.
- `BaseTracker._commit_causal_frame(frame_index)` advances only after a successful, fully validated prediction.
- `Tracker.create_tracker(params, mode=None)` preserves VOT's explicit mode and fixes OPE/video's omitted mode.

- [ ] **Step 1: Write the causal schema RED tests**

Create `tests/test_spectral_causality.py` with these imports and cases:

```python
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from lib.test.evaluation.causal import (
    CausalFrameRecord,
    assert_causal_tracker_info,
    call_causal_track,
)


class ForbiddenValueMapping(dict):
    def __getitem__(self, key):
        if key == "gt_bbox":
            raise AssertionError("forbidden value was read")
        return super().__getitem__(key)


class CausalSchemaTests(unittest.TestCase):
    def test_whitelist_does_not_read_forbidden_values(self):
        previous = ForbiddenValueMapping(
            target_bbox=[1, 2, 3, 4],
            best_score=0.75,
            gt_bbox=[9, 9, 9, 9],
        )
        record = CausalFrameRecord.from_evaluator(3, previous)
        safe = record.as_tracker_info()
        self.assertEqual(set(safe), {"frame_index", "previous_output"})
        self.assertEqual(
            set(safe["previous_output"]), {"target_bbox", "best_score"}
        )

    def test_values_are_copied_and_deeply_immutable(self):
        boxes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        record = CausalFrameRecord.from_evaluator(
            1,
            {
                "target_bbox": [1.0, 2.0, 3.0, 4.0],
                "all_boxes": boxes,
            },
        )
        first = record.as_tracker_info()["previous_output"]["all_boxes"]
        second = record.as_tracker_info()["previous_output"]["all_boxes"]
        self.assertEqual(first, (1.0, 2.0, 3.0, 4.0))
        self.assertIs(first, second)
        boxes.zero_()
        self.assertEqual(record.previous_output["all_boxes"], first)
        with self.assertRaises(TypeError):
            first[0] = 9.0

    def test_grad_tensors_and_invalid_shapes_fail_closed(self):
        with self.assertRaises(ValueError):
            CausalFrameRecord.from_evaluator(
                1, {"target_bbox": torch.ones(4, requires_grad=True)}
            )
        invalid = (
            {"target_bbox": []},
            {"target_bbox": [1, 2, 3]},
            {"target_bbox": [[1, 2, 3, 4]]},
            {"all_boxes": [1, 2, 3, 4, 5]},
            {"all_boxes": [[1, 2, 3, 4]]},
            {"all_scores": []},
            {"all_scores": [[0.5]]},
        )
        for previous in invalid:
            with self.subTest(previous=previous), self.assertRaises(
                (TypeError, ValueError)
            ):
                CausalFrameRecord.from_evaluator(1, previous)

    def test_direct_constructor_cannot_bypass_copy_and_validation(self):
        source = [1.0, 2.0, 3.0, 4.0]
        record = CausalFrameRecord(1, {"target_bbox": source})
        source[0] = 99.0
        self.assertEqual(
            record.as_tracker_info()["previous_output"]["target_bbox"],
            (1.0, 2.0, 3.0, 4.0),
        )
        with self.assertRaises(ValueError):
            CausalFrameRecord(1, {"target_bbox": [1, 2, 3]})
        with self.assertRaises(ValueError):
            CausalFrameRecord(1, {"best_score": 0.5})

    def test_call_helper_revalidates_record_before_dispatch(self):
        tracker = RecordingTracker()
        record = CausalFrameRecord(1, {"target_bbox": [1, 2, 3, 4]})
        call_causal_track(tracker, object(), record)
        self.assertEqual(len(tracker.track_infos), 1)

    def test_invalid_frame_and_prediction_values_fail_closed(self):
        for value in (True, 0, -1, 1.0, "1"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                CausalFrameRecord.from_evaluator(value, {})
        for value in (float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                CausalFrameRecord.from_evaluator(1, {"best_score": value})

    def test_unknown_keys_fail_at_both_levels(self):
        with self.assertRaises(ValueError):
            assert_causal_tracker_info({"frame_index": 1, "future": 2})
        with self.assertRaises(ValueError):
            assert_causal_tracker_info({
                "frame_index": 1,
                "previous_output": {"gt_bbox": [1, 2, 3, 4]},
            })

    def test_initialization_allows_only_one_valid_bbox(self):
        from lib.test.evaluation.causal import sanitize_initialization_info
        safe = sanitize_initialization_info({"init_bbox": [1, 2, 3, 4]})
        self.assertEqual(safe, {"init_bbox": [1.0, 2.0, 3.0, 4.0]})
        for invalid in (
            {"init_bbox": [1, 2, 3, 4], "object_ids": [1]},
            {"init_bbox": {1: [1, 2, 3, 4]}},
            {"init_bbox": [1, 2, 0, 4]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                sanitize_initialization_info(invalid)
```

- [ ] **Step 2: Add RED tests that hit production OPE/video/VOT paths**

In the same file add the following complete tests. The OPE test invokes real
`Tracker.run_sequence` with two search frames. The video test mocks only the
external GUI/capture boundary and executes real `Tracker.run_video` through an
`r` reset. History and local frame index are committed only after a successful
`track` return.

```python
class RecordingTracker:
    def __init__(self):
        self.params = SimpleNamespace(
            save_all_boxes=False,
            debug=0,
            multiobj_mode="default",
            tracker_name="seatrack",
        )
        self.begin_calls = []
        self.initialize_calls = []
        self.track_infos = []
        self.returned_outputs = []

    def begin_episode(self, reset_global=True):
        self.begin_calls.append(reset_global)

    def initialize(self, image, info):
        self.initialize_calls.append(dict(info))
        return None

    def track(self, image, *, info):
        assert_causal_tracker_info(info)
        self.track_infos.append(info)
        call_number = len(self.track_infos)
        output = {
            "target_bbox": [10.0 * call_number, 2.0, 3.0, 4.0],
            "best_score": 0.5,
        }
        self.returned_outputs.append(output)
        return output


class SequenceSentinel:
    def __init__(self):
        self.frames = ["frame0", "frame1", "frame2"]
        self.ground_truth_rect = object()
        self.multiobj_mode = False
        self.init_data = {0: {}}

    def init_info(self):
        return {"init_bbox": [1.0, 2.0, 3.0, 4.0]}

    def frame_info(self, frame_num):
        raise AssertionError("post-initialization frame_info is forbidden")


def bare_wrapper():
    from lib.test.evaluation.tracker import Tracker

    wrapper = Tracker.__new__(Tracker)
    wrapper.dataset_name = "synthetic"
    wrapper.tracker_class = RecordingTracker
    wrapper._read_image = lambda _: object()
    return wrapper


class ProductionEntryPointTests(unittest.TestCase):
    def test_real_run_sequence_is_causal_and_commits_last_prediction(self):
        tracker = RecordingTracker()
        wrapper = bare_wrapper()
        wrapper.get_parameters = lambda: tracker.params
        wrapper.create_tracker = lambda params: tracker
        reads = 0

        def read_image(_):
            nonlocal reads
            reads += 1
            if reads == 3:
                # Mutate the tracker-owned first return after evaluator commit
                # but before construction of the second frame's record.
                tracker.returned_outputs[0]["target_bbox"][0] = 999.0
            return object()

        wrapper._read_image = read_image
        output = wrapper.run_sequence(SequenceSentinel())
        self.assertEqual(tracker.begin_calls, [True])
        self.assertEqual(len(tracker.initialize_calls), 1)
        self.assertEqual(
            [info["frame_index"] for info in tracker.track_infos], [1, 2]
        )
        self.assertNotIn("gt_bbox", tracker.track_infos[0]["previous_output"])
        self.assertEqual(
            tracker.track_infos[1]["previous_output"]["target_bbox"],
            (10.0, 2.0, 3.0, 4.0),
        )
        self.assertEqual(len(output["target_bbox"]), 3)

    def test_ope_rejects_unsupported_protocols_before_image_read(self):
        for change in ("multiobj", "mid_sequence"):
            seq = SequenceSentinel()
            if change == "multiobj":
                seq.multiobj_mode = True
            else:
                seq.init_data = {0: {}, 2: {}}
            wrapper = bare_wrapper()
            wrapper._read_image = mock.Mock(side_effect=AssertionError("image read"))
            with self.subTest(change=change), self.assertRaises(ValueError):
                wrapper._track_sequence(RecordingTracker(), seq, seq.init_info())
            wrapper._read_image.assert_not_called()

    def test_create_tracker_defaults_mode_to_none(self):
        wrapper = bare_wrapper()
        seen = []

        class Factory:
            def __init__(self, params, mode):
                seen.append(mode)

        wrapper.tracker_class = Factory
        wrapper.create_tracker(object())
        self.assertEqual(seen, [None])

    def test_video_frame_helper_is_keyword_only(self):
        tracker = RecordingTracker()
        output, frozen_history = bare_wrapper()._track_video_frame(
            tracker,
            object(),
            frame_index=1,
            previous_output={"target_bbox": [1, 2, 3, 4]},
        )
        self.assertEqual(output["best_score"], 0.5)
        self.assertEqual(
            frozen_history["target_bbox"], (10.0, 2.0, 3.0, 4.0)
        )
        self.assertEqual(tracker.track_infos[0]["frame_index"], 1)

    def test_real_run_video_resets_history_and_frame_index_on_roi_reset(self):
        import lib.test.evaluation.tracker as tracker_module

        class FakeCapture:
            def __init__(self):
                self.frames = [
                    np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)
                ]
                self.released = False

            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

            def release(self):
                self.released = True

        tracker = RecordingTracker()
        wrapper = bare_wrapper()
        wrapper.name = "seatrack"
        wrapper.parameter_name = "rgbt"
        wrapper.get_parameters = lambda: tracker.params
        wrapper.create_tracker = lambda params: tracker
        capture = FakeCapture()
        with (
            mock.patch.object(tracker_module.os.path, "isfile", return_value=True),
            mock.patch.object(tracker_module.cv, "VideoCapture", return_value=capture),
            mock.patch.object(tracker_module.cv, "namedWindow"),
            mock.patch.object(tracker_module.cv, "resizeWindow"),
            mock.patch.object(tracker_module.cv, "imshow"),
            mock.patch.object(tracker_module.cv, "putText"),
            mock.patch.object(tracker_module.cv, "rectangle"),
            mock.patch.object(tracker_module.cv, "destroyAllWindows"),
            mock.patch.object(
                tracker_module.cv, "selectROI", return_value=(5, 6, 7, 8)
            ),
            mock.patch.object(
                tracker_module.cv, "waitKey",
                side_effect=[-1, ord("r"), ord("q")],
            ),
        ):
            wrapper.run_video("synthetic.mp4", optional_box=[1, 2, 3, 4])
        self.assertTrue(capture.released)
        self.assertEqual(tracker.begin_calls, [True, True])
        self.assertEqual(
            [info["frame_index"] for info in tracker.track_infos], [1, 2, 1]
        )
        self.assertEqual(
            tracker.track_infos[1]["previous_output"]["target_bbox"],
            (10.0, 2.0, 3.0, 4.0),
        )
        self.assertEqual(
            tracker.track_infos[2]["previous_output"]["target_bbox"],
            (5.0, 6.0, 7.0, 8.0),
        )

    def test_video_rejects_parallel_invalid_roi_and_initial_read_failure(self):
        import lib.test.evaluation.tracker as tracker_module

        wrapper = bare_wrapper()
        wrapper.name = "seatrack"
        wrapper.parameter_name = "rgbt"
        tracker = RecordingTracker()
        wrapper.get_parameters = lambda: tracker.params
        wrapper.create_tracker = lambda params: tracker
        tracker.params.multiobj_mode = "parallel"
        with self.assertRaises(ValueError):
            wrapper.run_video("unused.mp4")

        tracker.params.multiobj_mode = "default"
        with (
            mock.patch.object(tracker_module.os.path, "isfile", return_value=True),
            mock.patch.object(tracker_module.cv, "VideoCapture") as capture_cls,
            mock.patch.object(tracker_module.cv, "namedWindow") as named_window,
            mock.patch.object(tracker_module.cv, "resizeWindow") as resize_window,
            mock.patch.object(tracker_module.cv, "imshow") as imshow,
        ):
            capture_cls.return_value.read.return_value = (False, None)
            with self.assertRaises(RuntimeError):
                wrapper.run_video("synthetic.mp4", optional_box=[1, 2, 3, 4])
        named_window.assert_not_called()
        resize_window.assert_not_called()
        imshow.assert_not_called()

        with self.assertRaises(ValueError):
            wrapper._initialize_video_episode(
                tracker, np.zeros((4, 4, 3), dtype=np.uint8), [1, 2, 0, 4]
            )

    def test_malformed_outputs_fail_before_history_commit(self):
        class MalformedTracker(RecordingTracker):
            def initialize(self, image, info):
                return {"all_boxes": [1, 2, 3]}

            def track(self, image, *, info):
                return {
                    "target_bbox": [1, 2, 3, float("nan")],
                    "best_score": 0.5,
                }

        tracker = MalformedTracker()
        wrapper = bare_wrapper()
        with self.assertRaises(ValueError):
            wrapper._initialize_video_episode(
                tracker, np.zeros((4, 4, 3), dtype=np.uint8), [1, 2, 3, 4]
            )
        with self.assertRaises(ValueError):
            wrapper._track_video_frame(
                tracker,
                object(),
                frame_index=1,
                previous_output={"target_bbox": [1, 2, 3, 4]},
            )

        class MissingTargetTracker(RecordingTracker):
            def track(self, image, *, info):
                return {"best_score": 0.5}

        with self.assertRaises(ValueError):
            wrapper._track_video_frame(
                MissingTargetTracker(),
                object(),
                frame_index=1,
                previous_output={"target_bbox": [1, 2, 3, 4]},
            )

    def test_real_vot_adapter_resets_each_episode(self):
        from lib.test.vot.seatrack_class import SEATrack as VOTAdapter

        tracker = RecordingTracker()
        adapter = VOTAdapter.__new__(VOTAdapter)
        adapter.tracker = tracker
        image = np.zeros((8, 8, 6), dtype=np.uint8)
        adapter.initialize(image, [1, 2, 3, 4])
        adapter.track(image)
        adapter.track(image)
        adapter.initialize(image, [5, 6, 7, 8])
        adapter.track(image)
        self.assertEqual(tracker.begin_calls, [True, True])
        self.assertEqual(
            [info["frame_index"] for info in tracker.track_infos], [1, 2, 1]
        )
        self.assertEqual(
            tracker.track_infos[1]["previous_output"]["target_bbox"],
            (10.0, 2.0, 3.0, 4.0),
        )
        self.assertEqual(
            tracker.track_infos[2]["previous_output"]["target_bbox"],
            (5.0, 6.0, 7.0, 8.0),
        )

    def test_vot_missing_score_does_not_commit_frame_or_history(self):
        from lib.test.vot.seatrack_class import SEATrack as VOTAdapter

        class MissingScoreTracker(RecordingTracker):
            def track(self, image, *, info):
                assert_causal_tracker_info(info)
                return {"target_bbox": [1, 2, 3, 4]}

        adapter = VOTAdapter.__new__(VOTAdapter)
        adapter.tracker = MissingScoreTracker()
        image = np.zeros((8, 8, 6), dtype=np.uint8)
        adapter.initialize(image, [1, 2, 3, 4])
        history_before = adapter._previous_output
        with self.assertRaises(ValueError):
            adapter.track(image)
        self.assertEqual(adapter._frame_index, 0)
        self.assertIs(adapter._previous_output, history_before)

    def test_run_vot_exp_does_not_own_a_second_reset(self):
        from lib.test.vot.seatrack_class import run_vot_exp

        self.assertNotIn("begin_episode", inspect.getsource(run_vot_exp))
```

The real `run_video` test, not a source-string assertion, is the authority that
`_initialize_video_episode` and `_track_video_frame` are wired into the GUI loop.

- [ ] **Step 2b: Add RED tests against the real BaseTracker, SEATrack, and OSTrack classes**

Append the following. `sample_target` is stopped immediately after lifecycle
preparation, so these tests require no checkpoint or GPU. `ImageSentinel` proves
schema/frame validation happens before `image.shape` and therefore before crop or
network forward.

```python
class StopAfterLifecycle(Exception):
    pass


class ImageAccessed(Exception):
    pass


class ImageSentinel:
    @property
    def shape(self):
        raise ImageAccessed("image was accessed")


def real_tracker_cases():
    import lib.test.tracker.ostrack as ostrack_module
    import lib.test.tracker.seatrack as seatrack_module

    return (
        (seatrack_module, seatrack_module.SEATrack),
        (ostrack_module, ostrack_module.OSTrack),
    )


def raw_real_tracker(tracker_class):
    tracker = tracker_class.__new__(tracker_class)
    tracker.params = SimpleNamespace(template_factor=2.0, template_size=128)
    tracker._episode_pending_initialization = False
    tracker.state = [9.0, 9.0, 9.0, 9.0]
    tracker.frame_id = 7
    tracker.network = object()
    return tracker


class RealTrackerLifecycleTests(unittest.TestCase):
    def test_base_begin_requires_bool(self):
        from lib.test.tracker.basetracker import BaseTracker

        tracker = BaseTracker(SimpleNamespace())
        for invalid in (1, 0, None, "true"):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                tracker.begin_episode(invalid)

    def test_real_begin_clears_only_episode_state_and_not_network(self):
        for _, tracker_class in real_tracker_cases():
            tracker = raw_real_tracker(tracker_class)
            network = tracker.network
            tracker.begin_episode(True)
            self.assertIsNone(tracker.state)
            self.assertEqual(tracker.frame_id, 0)
            self.assertTrue(tracker._episode_pending_initialization)
            self.assertIs(tracker.network, network)

    def test_direct_initialize_fallback_begins_once_then_consumes_pending(self):
        image = np.zeros((8, 8, 6), dtype=np.uint8)
        info = {"init_bbox": [1, 2, 3, 4]}
        for tracker_module, tracker_class in real_tracker_cases():
            tracker = raw_real_tracker(tracker_class)
            calls = []
            real_begin = tracker.begin_episode

            def begin_spy(reset_global=True, real_begin=real_begin, calls=calls):
                calls.append(reset_global)
                return real_begin(reset_global)

            tracker.begin_episode = begin_spy
            with mock.patch.object(
                tracker_module, "sample_target", side_effect=StopAfterLifecycle
            ):
                with self.assertRaises(StopAfterLifecycle):
                    tracker.initialize(image, info)
            self.assertEqual(calls, [True])
            self.assertFalse(tracker._episode_pending_initialization)

    def test_explicit_begin_is_not_repeated_by_initialize(self):
        image = np.zeros((8, 8, 6), dtype=np.uint8)
        info = {"init_bbox": [1, 2, 3, 4]}
        for tracker_module, tracker_class in real_tracker_cases():
            tracker = raw_real_tracker(tracker_class)
            tracker.begin_episode(True)
            calls = []
            real_begin = tracker.begin_episode

            def begin_spy(reset_global=True, real_begin=real_begin, calls=calls):
                calls.append(reset_global)
                return real_begin(reset_global)

            tracker.begin_episode = begin_spy
            with mock.patch.object(
                tracker_module, "sample_target", side_effect=StopAfterLifecycle
            ):
                with self.assertRaises(StopAfterLifecycle):
                    tracker.initialize(image, info)
            self.assertEqual(calls, [])
            self.assertFalse(tracker._episode_pending_initialization)

    def test_real_initialize_rejects_noncausal_info_before_reset_or_crop(self):
        for tracker_module, tracker_class in real_tracker_cases():
            tracker = raw_real_tracker(tracker_class)
            old_state = list(tracker.state)
            with mock.patch.object(tracker_module, "sample_target") as sampler:
                with self.assertRaises(ValueError):
                    tracker.initialize(
                        object(),
                        {
                            "init_bbox": [1, 2, 3, 4],
                            "gt_bbox": [1, 2, 3, 4],
                        },
                    )
            sampler.assert_not_called()
            self.assertEqual(tracker.state, old_state)
            self.assertFalse(tracker._episode_pending_initialization)

    def test_real_trackers_fail_before_image_access_and_enforce_order(self):
        valid = {
            "frame_index": 1,
            "previous_output": {"target_bbox": [1, 2, 3, 4]},
        }
        for _, tracker_class in real_tracker_cases():
            tracker = raw_real_tracker(tracker_class)
            tracker.frame_id = 0
            tracker.state = None
            with self.assertRaises(RuntimeError):
                tracker.track(ImageSentinel(), info=valid)

            tracker.state = [1, 2, 3, 4]
            forbidden = {
                "frame_index": 1,
                "previous_output": {"gt_bbox": [1, 2, 3, 4]},
            }
            with self.assertRaises(ValueError):
                tracker.track(ImageSentinel(), info=forbidden)
            with self.assertRaises(ValueError):
                tracker.track(
                    ImageSentinel(),
                    info={
                        "frame_index": 2,
                        "previous_output": {
                            "target_bbox": [1, 2, 3, 4],
                        },
                    },
                )
            self.assertEqual(tracker.frame_id, 0)

            with self.assertRaises(ImageAccessed):
                tracker.track(ImageSentinel(), info=valid)
            self.assertEqual(tracker.frame_id, 0)

            source = inspect.getsource(tracker_class.track)
            self.assertLess(
                source.index("_validate_causal_frame"),
                source.index("image.shape"),
            )
            self.assertLess(
                source.index("freeze_prediction_history"),
                source.index("_commit_causal_frame"),
            )

            next_frame = tracker._validate_causal_frame(valid)
            self.assertEqual(next_frame, 1)
            tracker._commit_causal_frame(next_frame)
            self.assertEqual(tracker.frame_id, 1)
            with self.assertRaises(ValueError):
                tracker._validate_causal_frame(valid)

    def test_ostrack_track_source_has_no_gt_read(self):
        from lib.test.tracker.ostrack import OSTrack

        self.assertNotIn("gt_bbox", inspect.getsource(OSTrack.track))
```

- [ ] **Step 3: Run RED**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_causality -v
```

Expected: initial import fails because `causal.py` is missing. After adding only that file, production tests still fail on the required `mode`, GT/frame-info access, missing video helper, and VOT lifecycle.

- [ ] **Step 4: Implement the immutable whitelist**

Create `lib/test/evaluation/causal.py` with this structure:

```python
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


def freeze_prediction_history(
    previous_output, required_keys=("target_bbox",)
):
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
    CausalFrameRecord.from_evaluator(info["frame_index"], info["previous_output"])


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
```

The schema intentionally accepts only flat vectors. Current SEATrack and OSTrack
emit flat `target_bbox`/`all_boxes` lists; nested/ragged data fails closed and is
never flattened silently. Factory and public constructor share `__post_init__`,
and the dispatch helper revalidates before the production call.

- [ ] **Step 5: Wire real lifecycle and keyword calls**

Implement the following exact seams and ordering.

In `lib/test/evaluation/tracker.py` add:

```python
from collections.abc import Mapping
from lib.test.evaluation.causal import (
    CausalFrameRecord,
    call_causal_track,
    freeze_prediction_history,
    sanitize_initialization_info,
)


def create_tracker(self, params, mode=None):
    return self.tracker_class(params, mode)
```

At the beginning of `_track_sequence`, before image access, validate the
supported protocol and sanitize the only legal GT-derived value:

```python
if getattr(seq, "multiobj_mode", False):
    raise ValueError("only single-object evaluation is supported")
if not isinstance(getattr(seq, "init_data", None), Mapping):
    raise TypeError("seq.init_data must be a mapping")
if set(seq.init_data) != {0}:
    raise ValueError("only frame-zero initialization is supported")
init_info = sanitize_initialization_info(init_info)
```

Replace only the initialization/loop control in `_track_sequence`; retain the
existing output dictionary, timing, `_store_outputs`, RGB/RGB-X reads, and final
single-element cleanup:

```python
start_time = time.time()
tracker.begin_episode(reset_global=True)
out = tracker.initialize(image, init_info)
if out is None:
    out = {}
if not isinstance(out, Mapping):
    raise TypeError("tracker.initialize must return a mapping or None")

initial_history = {"target_bbox": list(init_info["init_bbox"])}
initial_history.update(out)
prev_output = freeze_prediction_history(initial_history)
init_default = {
    "target_bbox": init_info["init_bbox"],
    "time": time.time() - start_time,
    "all_scores": 1,
}
# Preserve the existing save_all_boxes branch here.
_store_outputs(out, init_default)

for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
    # Preserve the existing dataset-specific image read here.
    start_time = time.time()
    record = CausalFrameRecord.from_evaluator(frame_num, prev_output)
    out = call_causal_track(tracker, image, record)
    if not isinstance(out, Mapping):
        raise TypeError("tracker.track must return a mapping")
    frozen_output = freeze_prediction_history(out)
    # Commit only after type, key, shape, gradient, and finite validation.
    prev_output = frozen_output
    _store_outputs(out, {"time": time.time() - start_time})
```

Delete all post-initialization `seq.frame_info`, `ground_truth_rect`, and
positional `tracker.track(image, info)` access.

Add the two video helpers:

```python
def _initialize_video_episode(self, tracker, image, box):
    init_info = sanitize_initialization_info({"init_bbox": box})
    tracker.begin_episode(reset_global=True)
    out = tracker.initialize(image, init_info)
    if out is None:
        out = {}
    if not isinstance(out, Mapping):
        raise TypeError("tracker.initialize must return a mapping or None")
    initial_history = {"target_bbox": list(init_info["init_bbox"])}
    initial_history.update(out)
    previous_output = freeze_prediction_history(initial_history)
    return 0, previous_output, list(init_info["init_bbox"])


def _track_video_frame(self, tracker, image, frame_index, previous_output):
    record = CausalFrameRecord.from_evaluator(frame_index, previous_output)
    out = call_causal_track(tracker, image, record)
    if not isinstance(out, Mapping):
        raise TypeError("tracker.track must return a mapping")
    frozen_output = freeze_prediction_history(out)
    return out, frozen_output
```

In real `run_video` replace the protocol/setup and both initialization sites
with the following state transitions. Existing drawing and result serialization
remain unchanged around this control flow:

```python
multiobj_mode = getattr(
    params, "multiobj_mode", getattr(self.tracker_class, "multiobj_mode", "default")
)
if multiobj_mode != "default":
    raise ValueError("run_video supports only single-object default mode")
tracker = self.create_tracker(params)

if not os.path.isfile(videofilepath):
    raise ValueError("videofilepath must be an existing file")
cap = cv.VideoCapture(videofilepath)
success, frame = cap.read()
if not success or frame is None:
    cap.release()
    raise RuntimeError(f"failed to read first frame from {videofilepath}")

# Create/show the window only after the successful read.
frame_index = 0
previous_output = None
if optional_box is not None:
    frame_index, previous_output, init_state = self._initialize_video_episode(
        tracker, frame, optional_box
    )
    output_boxes.append(init_state)
else:
    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
    frame_index, previous_output, init_state = self._initialize_video_episode(
        tracker, frame, [x, y, w, h]
    )
    output_boxes.append(init_state)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    next_frame_index = frame_index + 1
    out, frozen_output = self._track_video_frame(
        tracker, frame, next_frame_index, previous_output
    )
    # Atomic commit before drawing/saving, after complete output validation.
    frame_index = next_frame_index
    previous_output = frozen_output

    # Keep existing prediction-only drawing/output code here.
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("r"):
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("failed to read reset frame")
        frame_disp = frame.copy()
        # Keep existing reset prompt/imshow here.
        x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
        frame_index, previous_output, init_state = self._initialize_video_episode(
            tracker, frame, [x, y, w, h]
        )
        output_boxes.append(init_state)
```

Remove the nested `_build_init_info`, every `assert`/`exit(-1)` in this path,
the undefined `MultiObjectWrapper` branch, and every direct
`tracker.initialize`/`tracker.track` call from `run_video`.

In `lib/test/tracker/basetracker.py` add the import and state-machine methods:

```python
from lib.test.evaluation.causal import assert_causal_tracker_info


class BaseTracker:
    def __init__(self, params):
        self.params = params
        self.visdom = None
        self._episode_pending_initialization = False

    def begin_episode(self, reset_global=True):
        if type(reset_global) is not bool:
            raise TypeError("reset_global must be bool")
        self._episode_pending_initialization = True

    def _prepare_episode_initialization(self):
        if not self._episode_pending_initialization:
            self.begin_episode(reset_global=True)
        self._episode_pending_initialization = False

    def _validate_causal_frame(self, info):
        if self._episode_pending_initialization or getattr(self, "state", None) is None:
            raise RuntimeError("tracker must be initialized before track")
        assert_causal_tracker_info(info)
        expected = self.frame_id + 1
        if info["frame_index"] != expected:
            raise ValueError(
                f"expected causal frame_index {expected}, got {info['frame_index']}"
            )
        return expected

    def _commit_causal_frame(self, frame_index):
        if frame_index != self.frame_id + 1:
            raise RuntimeError("causal frame commit is stale or out of order")
        self.frame_id = frame_index
```

In both `lib/test/tracker/seatrack.py` and
`lib/test/tracker/ostrack.py` import the causal output validator and
`sanitize_initialization_info`, then make the same lifecycle edits:

```python
from lib.test.evaluation.causal import (
    freeze_prediction_history,
    sanitize_initialization_info,
)


def begin_episode(self, reset_global=True):
    super().begin_episode(reset_global)
    self.state = None
    self.frame_id = 0


def initialize(self, image, info: dict):
    info = sanitize_initialization_info(info)
    self._prepare_episode_initialization()
    # Existing initialize arithmetic follows unchanged.


def track(self, image, info: dict = None):
    next_frame_index = self._validate_causal_frame(info)
    H, W, _ = image.shape
    # Existing track arithmetic follows unchanged, except delete the old:
    # self.frame_id += 1
```

For SEATrack, the last signature deliberately replaces
`track(self, image, dataset_name=None, save_name=None, seq_name=None, info=None)`;
repository search confirms those unused positional metadata arguments have no
callers. Do not reorder or edit crop, preprocess, forward, box-head, Hann,
or mapping arithmetic after the new validation line. Replace debug-only
`self.frame_id` reads with `next_frame_index` so filenames/titles retain their
existing visible numbering.

In SEATrack, replace the final direct-return branch exactly:

```python
if self.save_all_boxes:
    all_boxes = self.map_box_back_batch(
        pred_boxes * self.params.search_size / resize_factor, resize_factor
    )
    all_boxes_save = all_boxes.view(-1).tolist()
    result = {
        "target_bbox": self.state,
        "all_boxes": all_boxes_save,
        "best_score": max_score,
    }
else:
    result = {"target_bbox": self.state, "best_score": max_score}
freeze_prediction_history(
    result, required_keys=("target_bbox", "best_score")
)
self._commit_causal_frame(next_frame_index)
return result
```

In OSTrack, replace its final direct-return branch exactly:

```python
if self.save_all_boxes:
    all_boxes = self.map_box_back_batch(
        pred_boxes * self.params.search_size / resize_factor, resize_factor
    )
    all_boxes_save = all_boxes.view(-1).tolist()
    result = {"target_bbox": self.state, "all_boxes": all_boxes_save}
else:
    result = {"target_bbox": self.state}
freeze_prediction_history(result)
self._commit_causal_frame(next_frame_index)
return result
```

Do not commit `frame_id` if crop, preprocess, forward, box mapping, result
construction, or finite/shape validation raises. In OSTrack Visdom,
replace the GT tuple with prediction-only state:

```python
self.visdom.register((image, self.state), "Tracking", 1, "Tracking")
```

In `lib/test/vot/seatrack_class.py` import the causal helpers and replace the two
adapter methods exactly:

```python
from lib.test.evaluation.causal import (
    CausalFrameRecord,
    call_causal_track,
    freeze_prediction_history,
    sanitize_initialization_info,
)


def initialize(self, img_rgb, selection):
    init_info = sanitize_initialization_info({"init_bbox": list(selection)})
    self.H, self.W, _ = img_rgb.shape
    self.tracker.begin_episode(reset_global=True)
    out = self.tracker.initialize(img_rgb, init_info)
    if out is None:
        out = {}
    if not isinstance(out, dict):
        raise TypeError("tracker.initialize must return a dict or None")
    initial_history = {"target_bbox": list(init_info["init_bbox"])}
    initial_history.update(out)
    previous_output = freeze_prediction_history(initial_history)
    self._frame_index = 0
    self._previous_output = previous_output


def track(self, img_rgb):
    next_frame_index = self._frame_index + 1
    record = CausalFrameRecord.from_evaluator(
        next_frame_index, self._previous_output
    )
    outputs = call_causal_track(self.tracker, img_rgb, record)
    if not isinstance(outputs, dict):
        raise TypeError("tracker.track must return a dict")
    frozen_output = freeze_prediction_history(
        outputs, required_keys=("target_bbox", "best_score")
    )
    # Commit only after complete output validation and before VOT report.
    self._frame_index = next_frame_index
    self._previous_output = frozen_output
    return outputs["target_bbox"], outputs["best_score"]
```

`run_vot_exp` continues to call only `adapter.initialize`; do not add a second
`begin_episode` there.

- [ ] **Step 6: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_spectral_causality -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git diff --check
git add lib/test/evaluation/causal.py lib/test/evaluation/tracker.py \
  lib/test/tracker/basetracker.py lib/test/tracker/seatrack.py \
  lib/test/tracker/ostrack.py lib/test/vot/seatrack_class.py \
  tests/test_spectral_causality.py
git commit -m "fix: enforce causal tracker lifecycle"
```

Expected: all tests pass, no skips/expected failures are added, and the commit contains only the listed files.

### Task 2: Deeply Isolate Evaluation Config Loads

**Files:**

- Create: `tests/test_config_isolation.py`
- Modify: `lib/config/seatrack/config.py`
- Modify: `lib/test/parameter/seatrack.py`

**Interfaces:** `clone_default_cfg()` and `load_config(filename)` return fresh recursive EasyDict graphs derived from a private pristine import-time snapshot.

- [ ] **Step 1: Write the real RED leak test**

```python
import copy
import os
import unittest
from pathlib import Path
from unittest import mock

from lib.config.seatrack.config import cfg, clone_default_cfg
from lib.test.parameter import seatrack as parameter_module


class ConfigIsolationTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.env = type(
            "Env",
            (),
            {"prj_dir": str(self.repo), "save_dir": str(self.repo)},
        )()
        self.singleton_before = copy.deepcopy(cfg)

    def tearDown(self):
        # Restore the existing singleton even when an assertion fails.
        cfg.clear()
        cfg.update(copy.deepcopy(self.singleton_before))

    def test_parameter_loads_are_deeply_isolated(self):
        with mock.patch.object(
            parameter_module, "env_settings", return_value=self.env
        ):
            lift = parameter_module.parameters("rgbt_lifttrack_pilot")
            legacy = parameter_module.parameters("rgbt")
        self.assertIsNot(lift.cfg, legacy.cfg)
        self.assertTrue(lift.cfg.MODEL.BILIFT.ENABLED)
        self.assertFalse(legacy.cfg.MODEL.BILIFT.ENABLED)
        self.assertEqual(cfg, self.singleton_before)
        lift.cfg.MODEL.BILIFT.LAYERS.append(99)
        self.assertNotIn(99, legacy.cfg.MODEL.BILIFT.LAYERS)
        self.assertNotIn(99, clone_default_cfg().MODEL.BILIFT.LAYERS)

    def test_pristine_clone_ignores_a_deliberately_polluted_singleton(self):
        cfg.MODEL.BILIFT.ENABLED = True
        cfg.MODEL.BILIFT.LAYERS.append(999)
        fresh = clone_default_cfg()
        self.assertFalse(fresh.MODEL.BILIFT.ENABLED)
        self.assertNotIn(999, fresh.MODEL.BILIFT.LAYERS)
        with mock.patch.object(
            parameter_module, "env_settings", return_value=self.env
        ):
            legacy = parameter_module.parameters("rgbt")
        self.assertFalse(legacy.cfg.MODEL.BILIFT.ENABLED)
        self.assertNotIn(999, legacy.cfg.MODEL.BILIFT.LAYERS)

    def test_base_modalities_keep_all_ce_tokens_in_both_orders(self):
        with mock.patch.object(
            parameter_module, "env_settings", return_value=self.env
        ):
            first = parameter_module.parameters("rgbd")
            second = parameter_module.parameters("rgbt")
            third = parameter_module.parameters("rgbd")
        self.assertEqual(first.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1])
        self.assertEqual(second.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1])
        self.assertEqual(third.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1])
        self.assertIsNot(first.cfg, third.cfg)

    def test_checkpoint_and_task_selection_remain_environment_driven(self):
        with (
            mock.patch.object(
                parameter_module, "env_settings", return_value=self.env
            ),
            mock.patch.dict(
                os.environ,
                {
                    "SEATRACK_CHECKPOINT": "/tmp/sentinel-checkpoint.pth.tar",
                    "SEATRACK_TASK": "sentinel-task",
                },
            ),
        ):
            params = parameter_module.parameters("rgbt")
        self.assertEqual(
            params.checkpoint, "/tmp/sentinel-checkpoint.pth.tar"
        )
        self.assertEqual(params.task, "sentinel-task")
        self.assertEqual(params.search_size, params.cfg.TEST.SEARCH_SIZE)
```

Initial RED may be `ImportError` for `clone_default_cfg`. After adding only a
placeholder interface, the deliberate-singleton-pollution test must still fail
if the implementation copies current `cfg` instead of the private pristine
snapshot.

- [ ] **Step 2: Run RED**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_config_isolation -v
```

- [ ] **Step 3: Implement the pristine snapshot**

In `lib/config/seatrack/config.py` add `import copy`. Immediately after
`cfg.TEST.EPOCH = 500` (the last default assignment) and before any helper that
can load external YAML, capture:

```python
_DEFAULT_CFG = copy.deepcopy(cfg)


def clone_default_cfg():
    return copy.deepcopy(_DEFAULT_CFG)
```

Keep the existing `update_config_from_file` behavior for training compatibility.
Immediately after that function, add:

```python
def load_config(filename):
    local_cfg = clone_default_cfg()
    update_config_from_file(filename, base_cfg=local_cfg)
    return local_cfg
```

Never recapture `_DEFAULT_CFG`, never expose it, and never implement
`clone_default_cfg` as `copy.deepcopy(cfg)`.

In `lib/test/parameter/seatrack.py` retain `cfg` as a compatibility export but
import `load_config` and replace only the evaluation-config reads:

```python
from lib.config.seatrack.config import cfg, load_config


def parameters(yaml_name: str, epoch=None, variants=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    yaml_file = os.path.join(
        prj_dir, "experiments/seatrack/%s.yaml" % yaml_name
    )
    local_cfg = load_config(yaml_file)
    params.cfg = local_cfg
    if os.environ.get("SEATRACK_PRINT_CONFIG") == "1":
        print("test config: ", local_cfg)

    params.template_factor = local_cfg.TEST.TEMPLATE_FACTOR
    params.template_size = local_cfg.TEST.TEMPLATE_SIZE
    params.search_factor = local_cfg.TEST.SEARCH_FACTOR
    params.search_size = local_cfg.TEST.SEARCH_SIZE

    # Keep the existing checkpoint_override and yaml_name-prefix checkpoint
    # branches byte-for-byte unchanged. They remain environment/yaml-name
    # driven and must not read TEST.EPOCH or any new cfg field.
    checkpoint_override = os.environ.get("SEATRACK_CHECKPOINT")
    if checkpoint_override:
        params.checkpoint = checkpoint_override
    else:
        if yaml_name.startswith("rgbt"):
            params.checkpoint = os.path.join(
                save_dir, "checkpoints/rgbt/SEATrack_ep0060.pth.tar"
            )
        elif yaml_name.startswith("rgbd"):
            params.checkpoint = os.path.join(
                save_dir, "checkpoints/rgbd/SEATrack_ep0025.pth.tar"
            )
        elif yaml_name.startswith("rgbe"):
            params.checkpoint = os.path.join(
                save_dir, "checkpoints/rgbe/SEATrack_ep0045.pth.tar"
            )
        else:
            raise ValueError(f"Unsupported SEATrack yaml_name: {yaml_name}")

    params.save_all_boxes = False
    params.task = os.environ.get("SEATRACK_TASK", yaml_name)
    return params
```

Only cfg-derived TEST values and printing switch to `local_cfg`. Checkpoint
selection and `params.task` semantics do not change; no weight path is inferred
from `TEST.EPOCH`. This task does not refactor the training CLI or global
`update_config_from_file(filename)` callers.

- [ ] **Step 4: Verify GREEN and commit**

```bash
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_config_isolation -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest tests.test_bilift_integration tests.test_training_integrity -v
/home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
git diff --check
git add lib/config/seatrack/config.py lib/test/parameter/seatrack.py \
  tests/test_config_isolation.py
git commit -m "fix: isolate SEATrack evaluation configs"
```

Expected: focused and full suites pass; loading one experiment cannot change another or the module singleton.

## Self-Review Before Execution

```bash
! rg -n "T[B]D|T[O]DO|implement lat[e]r|fill in detail[s]|Similar to Tas[k]" \
  docs/superpowers/plans/2026-07-13-target-spectral-a0-causal-config.md
git add -N docs/superpowers/plans/2026-07-13-target-spectral-a0-causal-config.md
git diff --check -- docs/superpowers/plans/2026-07-13-target-spectral-a0-causal-config.md
git reset -- docs/superpowers/plans/2026-07-13-target-spectral-a0-causal-config.md
```

Expected: placeholder scan empty and whitespace check passes for the previously untracked plan. Reset removes only intent-to-add; it does not modify file content.
