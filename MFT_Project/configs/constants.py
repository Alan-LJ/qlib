"""Centralised configuration constants for the MFT project."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Commonly used filesystem locations within the project."""

    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    configs_dir: Path = root / "configs"
    dataset_dir: Path = root / "dataset"
    model_dir: Path = root / "model"
    explain_dir: Path = root / "explain"
    workflow_dir: Path = root / "workflow"
    scripts_dir: Path = root / "scripts"


@dataclass(frozen=True)
class DefaultExperimentWindow:
    """Default experiment date ranges for quick-start experiments."""

    train_start: str = "2023-01-02"
    train_end: str = "2023-03-31"
    valid_start: str = "2023-04-01"
    valid_end: str = "2023-05-15"
    test_start: str = "2023-05-16"
    test_end: str = "2023-06-30"


PATHS = ProjectPaths()
WINDOW = DefaultExperimentWindow()
