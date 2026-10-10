from pathlib import Path

from prismatic.local_paths import resolve_local_model_paths


MODEL_ROOT = Path("/ssd/wyzhang/all_models/EXPRESSS_models")


def test_prismatic_local_paths_point_to_downloaded_files():
    paths = resolve_local_model_paths(MODEL_ROOT)

    assert paths["prismatic_checkpoint"].is_dir()
    assert (paths["prismatic_checkpoint"] / "config.json").is_file()
    assert (paths["prismatic_checkpoint"] / "checkpoints" / "latest-checkpoint.pt").is_file()
    assert paths["llm_directory"].is_dir()
    assert paths["dino_checkpoint"].is_file()
    assert paths["siglip_checkpoint"].is_file()
