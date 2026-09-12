import os
import pathlib

import pytest

SCAN = pathlib.Path(os.environ.get("SCAN", "/ceph/christian/index_scanner"))
CHAIR_VIEWS = SCAN / "sessions/chairbase-dslr-1-all/demo/chair_views.vviews"


@pytest.fixture(scope="session")
def chair_views():
    """The ten-view chair demo set (skips where the scan data is absent)."""
    import volumetric

    if not CHAIR_VIEWS.exists():
        pytest.skip(f"{CHAIR_VIEWS} not present")
    return volumetric.ViewSet.load(str(CHAIR_VIEWS))
