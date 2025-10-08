import shutil
from pathlib import Path

import pytest

# ---------------- Session-level fixtures ----------------


@pytest.fixture(scope="session", autouse=True)
def prepare_and_cleanup_test_configs():
    """
    Copy all YAML configs from `tsfb_configs/` to `tests/test_configs/`
    before running tests, and remove the copied folder afterwards.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config_source = repo_root / "tsfb_configs"
    config_target = repo_root / "tests" / "test_configs"

    config_target.mkdir(parents=True, exist_ok=True)

    for ext in ("*.yml", "*.yaml"):
        for file in config_source.glob(ext):
            shutil.copy(file, config_target / file.name)

    # Yield control to the test session
    yield

    # Cleanup copied configs
    if config_target.exists():
        shutil.rmtree(config_target)


@pytest.fixture(autouse=True)
def set_repo_root_cwd_and_sys_path(request, monkeypatch):
    """
    Ensure the current working directory is the repository root so that
    `from tsfb.pipeline_executor import PipelineExecutor` works without
    installing the package. This mimics running from repo root in CI.
    """
    test_dir = Path(request.fspath.dirname)
    repo_root = test_dir.parent
    monkeypatch.chdir(repo_root)

    # Make sure repo root is on sys.path for module discovery
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return repo_root


# ---------------- Tests ----------------


class TestPipelineExecutor:
    def test_all_configs(self, tmp_path, caplog):
        """
        For each YAML config in `tests/test_configs/`:
         - Run the PipelineExecutor directly (no subprocess).
         - Verify an output CSV is produced and non-empty.
         - Remove the output CSV after each run to avoid cross-test interference.
        """
        caplog.set_level("INFO")
        repo_root = Path(__file__).resolve().parent.parent
        config_dir = repo_root / "tests" / "test_configs"

        # Collect configs
        config_files = sorted(
            list(config_dir.glob("*.yml")) + list(config_dir.glob("*.yaml"))
        )
        assert config_files, f"No YAML config files found in {config_dir}"

        for cfg_file in config_files:
            # Use a unique result file per config under a temp directory
            result_file = tmp_path / f"{cfg_file.stem}_results.csv"
            if result_file.exists():
                result_file.unlink()

            from tsfb.run_pipeline import PipelineExecutor

            # Run pipeline
            executor = PipelineExecutor(
                config_path=str(cfg_file), results_path=str(result_file)
            )
            try:
                executor.run()
            except Exception as e:
                pytest.fail(f"Pipeline failed for {cfg_file.name}: {e}")

            # Assertions on output
            assert result_file.exists(), f"Result file not created for {cfg_file.name}"
            assert (
                result_file.stat().st_size > 0
            ), f"Result file is empty for {cfg_file.name}"

            # Per-config cleanup
            result_file.unlink(missing_ok=True)

    def test_clean_up(self):
        """
        Remove common artifact directories that may be created during runs.
        Adjust the list if your project generates additional artifacts.
        """
        self._clean_folder()

    def _clean_folder(self):
        repo_root = Path(__file__).resolve().parent.parent
        paths = [
            repo_root / "experiments",
            # Add more artifact paths here if needed...
        ]
        for p in paths:
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def final_cleanup_session(request):
    """
    Final session cleanup hook. Add any last-resort cleanup here if your CI
    environment sometimes leaves behind extra files or folders.
    """
    repo_root = Path(__file__).resolve().parent.parent

    def _final():
        paths = [
            repo_root / "experiments",
            # Do NOT remove `tsfb_configs/` here because it's the real source.
            # We only remove `tests/test_configs/` in the other session fixture.
        ]
        for p in paths:
            if p.is_file():
                p.unlink(missing_ok=True)
            else:
                shutil.rmtree(p, ignore_errors=True)

    request.addfinalizer(_final)
