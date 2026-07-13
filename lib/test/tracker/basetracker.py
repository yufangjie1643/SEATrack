import time

import torch
from _collections import OrderedDict

from lib.train.data.processing_utils import transform_image_to_crop
from lib.test.evaluation.causal import assert_causal_tracker_info
from lib.vis.visdom_cus import Visdom


class BaseTracker:
    """Base class for all trackers."""

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

    def _commit_causal_frame(self, frame_index, next_state):
        if frame_index != self.frame_id + 1:
            raise RuntimeError("causal frame commit is stale or out of order")
        self.state = next_state
        self.frame_id = frame_index

    def predicts_segmentation_mask(self):
        return False

    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')

    def transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        # box_in: list [x1, y1, w, h], not normalized
        # box_extract: same as box_in
        # out bbox: Torch.tensor [1, 1, 4], x1y1wh, normalized
        if crop_type == 'template':
            crop_sz = torch.Tensor([self.params.template_size, self.params.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                                     visdom_info=visdom_info)
            except:
                time.sleep(0.5)
                print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
                      '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

            elif data['key'] == 'n':
                self.next_seq = True
