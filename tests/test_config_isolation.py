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
        # Restore the compatibility singleton even when a test assertion fails.
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
            for order in (("rgbd", "rgbt"), ("rgbt", "rgbd")):
                with self.subTest(order=order):
                    first = parameter_module.parameters(order[0])
                    second = parameter_module.parameters(order[1])
                    repeated = parameter_module.parameters(order[0])

                    self.assertEqual(
                        first.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1]
                    )
                    self.assertEqual(
                        second.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1]
                    )
                    self.assertEqual(
                        repeated.cfg.MODEL.BACKBONE.CE_KEEP_RATIO, [1, 1, 1]
                    )
                    self.assertIsNot(first.cfg, second.cfg)
                    self.assertIsNot(first.cfg, repeated.cfg)

        self.assertEqual(cfg, self.singleton_before)

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
        self.assertEqual(cfg, self.singleton_before)


if __name__ == "__main__":
    unittest.main()
