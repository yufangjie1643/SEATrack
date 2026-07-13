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

    def test_run_video_releases_resources_when_output_validation_fails(self):
        import lib.test.evaluation.tracker as tracker_module

        class MalformedTracker(RecordingTracker):
            def track(self, image, *, info):
                assert_causal_tracker_info(info)
                return {
                    "target_bbox": [1.0, 2.0, 3.0, float("nan")],
                    "best_score": 0.5,
                }

        class FakeCapture:
            def __init__(self):
                self.frames = [
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    np.zeros((8, 8, 3), dtype=np.uint8),
                ]
                self.released = False

            def read(self):
                if not self.frames:
                    return False, None
                return True, self.frames.pop(0)

            def release(self):
                self.released = True

        tracker = MalformedTracker()
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
            mock.patch.object(tracker_module.cv, "destroyAllWindows") as destroy,
        ):
            with self.assertRaises(ValueError):
                wrapper.run_video("synthetic.mp4", optional_box=[1, 2, 3, 4])
        self.assertTrue(capture.released)
        destroy.assert_called_once_with()

    def test_run_video_releases_resources_when_reset_read_fails(self):
        import lib.test.evaluation.tracker as tracker_module

        class FakeCapture:
            def __init__(self):
                self.frames = [
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    np.zeros((8, 8, 3), dtype=np.uint8),
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
            mock.patch.object(tracker_module.cv, "waitKey", return_value=ord("r")),
            mock.patch.object(tracker_module.cv, "destroyAllWindows") as destroy,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to read reset frame"):
                wrapper.run_video("synthetic.mp4", optional_box=[1, 2, 3, 4])
        self.assertTrue(capture.released)
        destroy.assert_called_once_with()

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
            next_state = [2.0, 3.0, 4.0, 5.0]
            tracker._commit_causal_frame(next_frame, next_state)
            self.assertEqual(tracker.frame_id, 1)
            self.assertEqual(tracker.state, next_state)
            with self.assertRaises(ValueError):
                tracker._validate_causal_frame(valid)

    def test_real_trackers_do_not_half_commit_when_output_freeze_fails(self):
        class FreezeFailure(Exception):
            pass

        class FakeBoxHead:
            def cal_bbox(self, response, size_map, offset_map, return_score=False):
                boxes = torch.tensor([[0.8, 0.7, 0.2, 0.2]])
                if return_score:
                    return boxes, torch.tensor([[0.75]])
                return boxes

        class FakeNetwork:
            def __init__(self):
                self.box_head = FakeBoxHead()

            def forward(self, **kwargs):
                return {
                    "score_map": torch.ones(1),
                    "size_map": torch.ones(1),
                    "offset_map": torch.ones(1),
                }

        class FakePreprocessor:
            def process(self, *args):
                return SimpleNamespace(tensors=torch.ones(1))

        image = np.zeros((100, 100, 6), dtype=np.uint8)
        info = {
            "frame_index": 1,
            "previous_output": {"target_bbox": [10, 20, 30, 40]},
        }
        for tracker_module, tracker_class in real_tracker_cases():
            with self.subTest(tracker=tracker_class.__name__):
                tracker = tracker_class.__new__(tracker_class)
                tracker.params = SimpleNamespace(
                    search_factor=2.0,
                    search_size=100,
                )
                tracker._episode_pending_initialization = False
                tracker.state = [10.0, 20.0, 30.0, 40.0]
                old_state = list(tracker.state)
                tracker.frame_id = 0
                tracker.save_all_boxes = True
                tracker.debug = 0
                tracker.use_visdom = False
                tracker.preprocessor = FakePreprocessor()
                tracker.network = FakeNetwork()
                tracker.output_window = torch.ones(1)
                tracker.box_mask_z = None
                tracker.z_tensor = torch.ones(1)
                tracker.z_dict1 = SimpleNamespace(tensors=torch.ones(1))
                captured = {}

                def reject_output(result, *args, **kwargs):
                    captured["result"] = result
                    raise FreezeFailure

                with (
                    mock.patch.object(
                        tracker_module,
                        "sample_target",
                        return_value=(image, 2.0, None),
                    ),
                    mock.patch.object(
                        tracker_module,
                        "clip_box",
                        side_effect=lambda box, *_args, **_kwargs: list(box),
                    ),
                    mock.patch.object(
                        tracker_module,
                        "freeze_prediction_history",
                        side_effect=reject_output,
                    ),
                ):
                    with self.assertRaises(FreezeFailure):
                        tracker.track(image, info=info)

                self.assertEqual(tracker.state, old_state)
                self.assertEqual(tracker.frame_id, 0)
                self.assertEqual(
                    captured["result"]["target_bbox"],
                    [35.0, 45.0, 10.0, 10.0],
                )
                self.assertTrue(
                    torch.allclose(
                        torch.tensor(captured["result"]["all_boxes"]),
                        torch.tensor([50.0, 55.0, 10.0, 10.0]),
                    )
                )
                source = inspect.getsource(tracker_class.track)
                self.assertLess(
                    source.index("freeze_prediction_history"),
                    source.index("_commit_causal_frame"),
                )

    def test_ostrack_track_source_has_no_gt_read(self):
        from lib.test.tracker.ostrack import OSTrack

        self.assertNotIn("gt_bbox", inspect.getsource(OSTrack.track))


if __name__ == "__main__":
    unittest.main()
