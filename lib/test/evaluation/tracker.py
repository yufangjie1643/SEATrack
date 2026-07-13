import importlib
import os
import pdb
from collections.abc import Mapping
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.causal import (
    CausalFrameRecord,
    call_causal_track,
    freeze_prediction_history,
    sanitize_initialization_info,
)
import time
import cv2 as cv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from lib.train.dataset.depth_utils import get_rgbd_frame


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params, mode=None):
        tracker = self.tracker_class(params, mode)  # self.dataset_name
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        if getattr(seq, 'multiobj_mode', False):
            raise ValueError("only single-object evaluation is supported")
        if not isinstance(getattr(seq, 'init_data', None), Mapping):
            raise TypeError("seq.init_data must be a mapping")
        if set(seq.init_data) != {0}:
            raise ValueError("only frame-zero initialization is supported")
        init_info = sanitize_initialization_info(init_info)

        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        '''
        add all_scores
        '''
        output = {'target_bbox': [],
                  'time': [],
                  'all_scores': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        if self.dataset_name in ['depthtrack_rgbd', 'depthtrack_rgbd_train', 'cdtb', 'votrgbd22_rgbd']:
            image = self._read_rgbd_image(seq.frames[0], dtype='rgbcolormap')
        else:
            image = self._read_image(seq.frames[0])

        start_time = time.time()
        tracker.begin_episode(reset_global=True)
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}
        if not isinstance(out, Mapping):
            raise TypeError("tracker.initialize must return a mapping or None")

        initial_history = {'target_bbox': list(init_info['init_bbox'])}
        initial_history.update(out)
        prev_output = freeze_prediction_history(initial_history)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'all_scores': 1}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            if self.dataset_name in ['depthtrack_rgbd', 'depthtrack_rgbd_train', 'cdtb', 'votrgbd22_rgbd']:
                image = self._read_rgbd_image(frame_path, dtype='rgbcolormap')
            else:
                image = self._read_image(frame_path)

            start_time = time.time()

            record = CausalFrameRecord.from_evaluator(frame_num, prev_output)
            out = call_causal_track(tracker, image, record)
            if not isinstance(out, Mapping):
                raise TypeError("tracker.track must return a mapping")
            frozen_output = freeze_prediction_history(out)
            prev_output = frozen_output
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def _initialize_video_episode(self, tracker, image, box):
        init_info = sanitize_initialization_info({'init_bbox': box})
        tracker.begin_episode(reset_global=True)
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}
        if not isinstance(out, Mapping):
            raise TypeError("tracker.initialize must return a mapping or None")
        initial_history = {'target_bbox': list(init_info['init_bbox'])}
        initial_history.update(out)
        previous_output = freeze_prediction_history(initial_history)
        return 0, previous_output, list(init_info['init_bbox'])

    def _track_video_frame(self, tracker, image, frame_index, previous_output):
        record = CausalFrameRecord.from_evaluator(frame_index, previous_output)
        out = call_causal_track(tracker, image, record)
        if not isinstance(out, Mapping):
            raise TypeError("tracker.track must return a mapping")
        frozen_output = freeze_prediction_history(out)
        return out, frozen_output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(
            params,
            'multiobj_mode',
            getattr(self.tracker_class, 'multiobj_mode', 'default'),
        )
        if multiobj_mode != 'default':
            raise ValueError("run_video supports only single-object default mode")
        tracker = self.create_tracker(params)
        if not os.path.isfile(videofilepath):
            raise ValueError("videofilepath must be an existing file")
        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        try:
            success, frame = cap.read()
            if not success or frame is None:
                raise RuntimeError(f"failed to read first frame from {videofilepath}")

            display_name = 'Display: ' + tracker.params.tracker_name
            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)
            cv.imshow(display_name, frame)

            frame_index = 0
            previous_output = None
            if optional_box is not None:
                frame_index, previous_output, init_state = self._initialize_video_episode(
                    tracker, frame, optional_box
                )
                output_boxes.append(init_state)
            else:
                frame_disp = frame.copy()
                cv.putText(
                    frame_disp,
                    'Select target ROI and press ENTER',
                    (20, 30),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (0, 0, 0),
                    1,
                )
                x, y, w, h = cv.selectROI(
                    display_name, frame_disp, fromCenter=False
                )
                frame_index, previous_output, init_state = self._initialize_video_episode(
                    tracker, frame, [x, y, w, h]
                )
                output_boxes.append(init_state)

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_disp = frame.copy()

                # Draw box
                next_frame_index = frame_index + 1
                out, frozen_output = self._track_video_frame(
                    tracker, frame, next_frame_index, previous_output
                )
                frame_index = next_frame_index
                previous_output = frozen_output
                state = [int(s) for s in out['target_bbox']]
                output_boxes.append(state)

                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

                font_color = (0, 0, 0)
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)

                # Display the resulting frame
                cv.imshow(display_name, frame_disp)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                if key == ord('r'):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise RuntimeError("failed to read reset frame")
                    frame_disp = frame.copy()

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                               (0, 0, 0), 1)

                    cv.imshow(display_name, frame_disp)
                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    frame_index, previous_output, init_state = self._initialize_video_episode(
                        tracker, frame, [x, y, w, h]
                    )
                    output_boxes.append(init_state)
        finally:
            try:
                cap.release()
            finally:
                cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

    def _read_rgbd_image(self, image_file: str, dtype='color'):
        if isinstance(image_file, dict):
            img = get_rgbd_frame(image_file['color'], image_file['depth'], dtype=dtype, depth_clip=True)
        else:
            if dtype == 'color':
                img = get_rgbd_frame(image_file, None, dtype=dtype, depth_clip=False)
            else:
                img = get_rgbd_frame(None, image_file, dtype=dtype, depth_clip=False)

        return img



