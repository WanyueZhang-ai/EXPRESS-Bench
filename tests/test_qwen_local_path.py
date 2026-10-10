from pathlib import Path

from gpt import QWEN3_VL_MODEL_PATH


def test_qwen3_vl_path_is_local_model_directory():
    model_path = Path(QWEN3_VL_MODEL_PATH)
    assert model_path.is_dir()
    assert (model_path / "config.json").is_file()
