import unittest


class QwenDeviceTests(unittest.TestCase):
    def test_explicit_worker_device_uses_single_device_map(self):
        from gpt import _device_map_for

        self.assertEqual(_device_map_for("cuda:3"), {"": "cuda:3"})
        self.assertEqual(_device_map_for("cpu"), {"": "cpu"})

    def test_auto_is_only_used_when_no_device_is_requested(self):
        from gpt import _device_map_for

        self.assertEqual(_device_map_for(None), "auto")
        self.assertEqual(_device_map_for(""), "auto")
