import json
from pathlib import Path

import pytest

from tests.fixture_export import build_export


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = build_export(tmp_path_factory.mktemp("export") / "synthetic_export")
    (root / ".paw").mkdir()
    (root / ".paw/episode_mapping.json").write_text('{"0":100,"1":101}')
    (root / "meta/pokeandwiggle_annotations.json").write_text(
        json.dumps(
            {
                "version": 1,
                "episodes": {
                    str(index): {"subtasks": [], "actors": []}
                    for index in range(2)
                },
            }
        )
    )
    return root


@pytest.fixture(scope="session")
def cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("paw-baseline-cache")
