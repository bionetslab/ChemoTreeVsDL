from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunPaths:
    """Container for all experiment-related directory paths."""
    training: Path
    results: Path
    folds: Path


def build_training_dir(base: Path, cohort: str) -> Path:
    """Build the ML feature matrices directory: one folder per cohort, settings are stored in config.json inside."""
    return base / "ml_feature_matrices" / cohort


def build_results_dir(base: Path, cohort: str, model_type: str) -> Path:
    """Build the ML results directory: one folder per model, run settings are stored in results_final.csv inside."""
    return base / "results" / cohort / "non_time_series" / model_type



def build_paths(
    base: Path,
    cohort: str,
    model_type: str,
    prefix: str = None,
    random_seed: int = None
) -> RunPaths:
    """
    Build the directory paths for an ML experiment.

    Layout: <base>/results/<cohort>/non_time_series/<model_type>/[seed_<n>/]<prefix>/
    The results directory is not created here (only training runs write to it).
    
    Args:
        base: Base directory for all experiments
        cohort: Name of the cohort
        model_type: ML model type (e.g. random_forest)
        prefix: Optional prefix for organizing runs (default: None)
        random_seed: Optional random seed for reproducibility (default: None)
        
    Returns:
        RunPaths object containing all directory paths
        
    Raises:
        OSError: If directory creation fails
    """
    
    if random_seed:
        base = base / f"multi_seed_training"
        
        
    training = build_training_dir(base, cohort)
    results = build_results_dir(base, cohort, model_type)
    folds = base / "folds" / cohort 
    
    if random_seed:
        training = training / f"seed_{random_seed}"
        results = results / f"seed_{random_seed}"
        folds = folds / f"seed_{random_seed}"
        
    if prefix:
        results = results / prefix
        

        
    # Create all directories atomically
    try:
        training.mkdir(parents=True, exist_ok=True)
        folds.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directories: {e}") from e
    
    return RunPaths(training=training, results=results, folds=folds)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    Atomically write bytes to a file using a temporary file.
    
    Args:
        path: Target file path
        data: Bytes data to write
        
    Raises:
        OSError: If write operation fails
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    except OSError as e:
        # Clean up temporary file if it exists
        if tmp_path.exists():
            tmp_path.unlink()
        raise OSError(f"Failed to write file {path}: {e}") from e


def save_pickle(obj: Any, path: Path) -> None:
    """
    Save an object to a pickle file atomically.
    
    Args:
        obj: Object to serialize
        path: Target file path
        
    Raises:
        OSError: If save operation fails
        pickle.PicklingError: If object cannot be pickled
    """
    try:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write_bytes(path, data)
    except pickle.PicklingError as e:
        raise pickle.PicklingError(f"Failed to pickle object: {e}") from e


def load_pickle(path: Path) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Deserialized object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If file cannot be unpickled
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to unpickle file {path}: {e}") from e


def fold_file(run_dir: Path, name: str, fold: int) -> Path:
    """
    Generate a standardized fold file path.
    
    Args:
        run_dir: Base directory for the run
        name: Base name for the file
        fold: Fold number
        
    Returns:
        Path to the fold file
    """
    return run_dir / f"{name}__fold_{fold}.pkl"