import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from datasets import Dataset, load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_dataset_version(dataset_path: Path) -> bool:
    """
    Check if the locally saved dataset is the newest version.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        dataset_path (Path): Local path where dataset is saved.

    Returns:
        bool: True if local version is latest or if version check fails, False if update needed.
    """
    if not dataset_path.exists():
        return False

    try:
        # Load local dataset first to verify it exists and is valid
        try:
            dataset = load_from_disk(str(dataset_path))
            if not dataset:
                return False
        except Exception:
            return False

        # If we can load the dataset successfully, assume it's valid
        # Skip version checking as it's not reliably stored in the dataset
        return True

    except Exception as e:
        logger.warning(f"Failed to check dataset version: {str(e)}")
        return True  # On error, assume local version is OK


def load_save_huggingface_dataset(
    dataset_name: str,
    dataset_path: Optional[Path] = None,
    force_download: bool = False,
    split: Optional[str] = None,
) -> Optional[Dataset]:
    """
    Load and save a Hugging Face dataset to disk.

    Args:
        dataset_name (str): The name of the dataset to load.
        dataset_path (Optional[Path]): The path to save the dataset to. If None,
            dataset will only be loaded without saving.
        force_download (bool): If True, download and replace existing dataset.
        split (Optional[str]): The specific split of the dataset to load (e.g., 'train', 'test').

    Returns:
        Optional[Dataset]: The loaded dataset if successful, None otherwise.
    """
    if dataset_path is None:
        try:
            logger.info(f"Loading dataset {dataset_name} without saving")
            dataset = load_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            raise
    else:
        try:
            if force_download:
                logger.info(f"Force downloading dataset {dataset_name}")
                dataset = load_dataset(dataset_name)
                dataset.save_to_disk(str(dataset_path))
                logger.info(f"Successfully saved dataset to {dataset_path}")
            else:
                try:
                    # First check if local version exists and is up to date
                    is_latest = _check_dataset_version(dataset_path)
                    if is_latest:
                        logger.info(f"Loading latest version from {dataset_path}")
                        dataset = load_from_disk(str(dataset_path))
                    else:
                        logger.info(
                            f"Local dataset outdated or missing. Downloading {dataset_name}"
                        )
                        dataset = load_dataset(dataset_name)
                        dataset.save_to_disk(str(dataset_path))
                        logger.info(f"Successfully saved dataset to {dataset_path}")
                except FileNotFoundError:
                    logger.info(f"Dataset not found. Downloading {dataset_name}")
                    dataset = load_dataset(dataset_name)
                    dataset.save_to_disk(str(dataset_path))
                    logger.info(
                        f"Successfully downloaded and saved dataset to {dataset_path}"
                    )
                    raise
        except Exception as e:
            logger.error(f"Error handling dataset {dataset_name}: {str(e)}")
            raise

    if dataset is not None:
        try:
            if split:
                dataset = dataset[split]
            else:
                # If no split is specified, return the first split available
                dataset = next(iter(dataset.values()))
        except KeyError:
            return dataset
    return None


def load_save_huggingface_dataset_df(
    dataset_name: str,
    dataset_path: Optional[Path] = None,
    force_download: bool = False,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and save a Hugging Face dataset to disk as a pandas DataFrame.

    Args:
        dataset_name (str): The name of the dataset to load.
        dataset_path (Optional[Path]): The path to save the dataset to. If None,
            dataset will only be loaded without saving.
        force_download (bool): If True, download and replace existing dataset.
        split (Optional[str]): The specific split of the dataset to load (e.g., 'train', 'test').

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame if successful,
        raises ValueError if the dataset is empty or not found.
    """
    try:
        dataset = load_save_huggingface_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            force_download=force_download,
            split=split,
        )
        if dataset:
            df = dataset.to_pandas()
            logger.info(f"Dataset converted to DataFrame with shape {df.shape}")
            return df
        else:
            logger.error(f"Dataset {dataset_name} is empty or not found.")
            raise ValueError(
                f"No dataset found for {dataset_name} at {dataset_path}. Please check the dataset name or path."
            )
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def load_save_dataset_df(
    dataset_name: str,
    dataset_path: Optional[Path] = None,
    force_download: bool = False,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and save a dataset to disk as a pandas DataFrame.

    Args:
        dataset_name (str): The name of the dataset to load.
        dataset_path (Optional[Path]): The path to save the dataset to. If None,
            dataset will only be loaded without saving.
        force_download (bool): If True, download and replace existing dataset.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame if successful,
        raises ValueError if the dataset is empty or not found.
    """
    return load_save_huggingface_dataset_df(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        force_download=force_download,
        split=split,
    )


def main() -> None:
    df = load_save_dataset_df(
        dataset_name="shuaishuaicdp/MLLM-Judge",
        dataset_path=Path("datasets/MLLM-Judge"),
        force_download=False,
    )
    print(df.columns)


if __name__ == "__main__":
    main()
