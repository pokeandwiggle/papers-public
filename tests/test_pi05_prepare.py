import json

import numpy as np
import pytest
from paw_baseline.pi05.geometry import Pi05Representation
from paw_baseline.pi05.prepare import chunk_action_values, prepare_pi05
from paw_baseline.prepare import prepare_dataset

from tests.fixture_export import build_export
from tests.test_pi05_geometry import physical


def test_stats_use_full_episode_chunks_and_pad_only_at_episode_end():
    action = np.stack([physical()] * 6)
    action[:, 0] = [0, 1, 2, 100, 101, 102]
    values = chunk_action_values(action, np.zeros(6), {0}, 3)
    np.testing.assert_allclose(
        values[:, 0],
        [0, 1, 2, 1, 2, 100, 2, 100, 101, 100, 101, 102, 101, 102, 102, 102, 102, 102],
        atol=1e-5,
    )


def test_prepared_view_provenance_and_training_only_stats(tmp_path):
    source = build_export(tmp_path / "source", episodes=2)
    base = prepare_dataset(
        source,
        cache_root=tmp_path / "cache",
        normalization_episodes=[0],
        visual_stats_profile="imagenet",
    )
    before = (base.root / "meta/stats.json").read_bytes()
    rep = Pi05Representation()
    prepared = prepare_pi05(base, 8)
    stamp = json.loads((prepared.root / ".paw/prepare.json").read_text())
    assert stamp["pi05"]["representation"] == rep.to_dict()
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [48 * 8]
    assert stats["observation.state"]["count"] == [48]
    assert (base.root / "meta/stats.json").read_bytes() == before
    assert prepare_pi05(base, 8) == prepared
    assert prepare_pi05(base, 9).root != prepared.root


def test_prepared_checkpoint_processors_and_serving_contract(tmp_path, monkeypatch):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
    )
    from lerobot.processor import PolicyProcessorPipeline
    from paw_baseline.layout import CAMERA_KEYS
    from paw_baseline.pi05.configuration_paw_pi05 import PawPI05Config
    from paw_baseline.pi05.processor_paw_pi05 import (
        PadNormalizedState,
        make_paw_pi05_pre_post_processors,
    )
    from paw_baseline.serving_metadata import (
        load_serving_metadata,
        metadata_from_prepared,
        write_serving_metadata,
    )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    )
    source = build_export(tmp_path / "source", episodes=2)
    base = prepare_dataset(source, cache_root=tmp_path / "cache")
    prepared = prepare_pi05(base, 8)
    config = PawPI05Config(
        device="cpu",
        chunk_size=8,
        n_action_steps=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (20,)),
            **{
                key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
                for key in CAMERA_KEYS
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (20,))},
    )
    config.validate_features()
    stats = json.loads((prepared.root / "meta/stats.json").read_text())
    pre, post = make_paw_pi05_pre_post_processors(config, stats)
    checkpoint = tmp_path / "checkpoint"
    config.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    loaded = PolicyProcessorPipeline.from_pretrained(
        checkpoint, config_filename="policy_preprocessor.json"
    )
    index = next(
        i for i, step in enumerate(loaded.steps) if isinstance(step, PadNormalizedState)
    )
    assert isinstance(loaded.steps[index + 1], Pi05PrepareStateTokenizerProcessorStep)
    metadata = metadata_from_prepared(prepared.root, model="pi05")
    write_serving_metadata(checkpoint, metadata)
    assert load_serving_metadata(checkpoint) == metadata
    config.chunk_size = 9
    config.save_pretrained(checkpoint)
    with pytest.raises(RuntimeError, match="disagree"):
        load_serving_metadata(checkpoint)
