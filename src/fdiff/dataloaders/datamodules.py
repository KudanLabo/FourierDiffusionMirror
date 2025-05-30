import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.fdiff.utils.dataclasses import collate_batch
from src.fdiff.utils.fourier import dft, localization_metrics, smooth_frequency
from src.fdiff.utils.preprocessing import (
    droughts_preprocess,
    mimic_preprocess,
    nasa_preprocess,
    nasdaq_preprocess,
)


class DiffusionDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        fourier_transform: bool = False,
        standardize: bool = False,
        X_ref: Optional[torch.Tensor] = None,
    ) -> None:
        """Dataset for diffusion models.

        Args:
            X (torch.Tensor): Time series that are fed to the model.
            y (Optional[torch.Tensor], optional): Potential labels. Defaults to None.
            fourier_transform (bool, optional): Performs a Fourier transform on the time series. Defaults to False.
            standardize (bool, optional): Standardize each feature in the dataset. Defaults to False.
            X_ref (Optional[torch.Tensor], optional): Features used to compute the mean and std. Defaults to None.
        """
        super().__init__()
        if fourier_transform:
            X = dft(X).detach()
        self.X = X
        self.y = y
        self.standardize = standardize
        if X_ref is None:
            X_ref = X
        elif fourier_transform:
            X_ref = dft(X_ref).detach()
        assert isinstance(X_ref, torch.Tensor)
        self.feature_mean = X_ref.mean(dim=0)
        self.feature_std = X_ref.std(dim=0)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = {}
        data["X"] = self.X[index]
        if self.standardize:
            data["X"] = (data["X"] - self.feature_mean) / self.feature_std
        if self.y is not None:
            data["y"] = self.y[index]
        return data


class Datamodule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__()
        # Cast data_dir to Path type
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir / self.dataset_name
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.fourier_transform = fourier_transform
        self.standardize = standardize
        self.X_train = torch.Tensor()
        self.y_train: Optional[torch.Tensor] = None
        self.X_test = torch.Tensor()
        self.y_test: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
            os.makedirs(self.data_dir)
            self.download_data()

    @abstractmethod
    def download_data(self) -> None:
        """Download the data."""
        ...

    def train_dataloader(self) -> DataLoader:
        train_set = DiffusionDataset(
            X=self.X_train,
            y=self.y_train,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
        )
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(
            X=self.X_test, y=self.y_test, fourier_transform=self.fourier_transform
        )
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(
            X=self.X_test,
            y=self.y_test,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
            X_ref=self.X_train,
        )
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    @abstractproperty
    def dataset_name(self) -> str: ...

    @property
    def dataset_parameters(self) -> dict[str, Any]:
        return {
            "n_channels": self.X_train.size(2),
            "max_len": self.X_train.size(1),
            "num_training_steps": len(self.train_dataloader()),
        }

    @property
    def feature_mean_and_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        train_set = DiffusionDataset(
            X=self.X_train,
            y=self.y_train,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
        )
        return train_set.feature_mean, train_set.feature_std


class ECGDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        subsample_localization: bool = False,
        smooth_frequency: bool = False,
        smoother_width: float = 0.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.subsample_localization = subsample_localization
        self.smooth_frequency = smooth_frequency
        self.smoother_width = smoother_width

    def setup(self, stage: str = "fit") -> None:
        # Read CSV; extract features and labels
        path_train = self.data_dir / "mitbih_train.csv"
        path_test = self.data_dir / "mitbih_test.csv"

        # Read data
        df_train = pd.read_csv(path_train)
        X_train = df_train.iloc[:, :187].values
        y_train = df_train.iloc[:, 187].values
        df_test = pd.read_csv(path_test)
        X_test = df_test.iloc[:, :187].values
        y_test = df_test.iloc[:, 187].values

        # Convert to tensor
        self.X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        # In case of subsampling, we only keep the time series that are most localized in time
        if self.subsample_localization:
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            loc_score = X_loc / X_spec_loc
            idx_ranking = torch.argsort(loc_score, descending=False)
            self.X_train = self.X_train[idx_ranking[:1000]]
            self.y_train = self.y_train[idx_ranking[:1000]]
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            logging.info("Subsampling the training set based on localization metrics.")
            logging.info(f"New time delocalization: {X_loc.mean().item():.3g}")
            logging.info(
                f"New frequency delocalization: {X_spec_loc.mean().item():.3g}"
            )

        # In case of frequency convolution, we convolve the frequency domain with a Gaussian kernel
        if self.smooth_frequency and self.smoother_width > 0.0:
            self.X_train = smooth_frequency(self.X_train, sigma=self.smoother_width)
            self.X_test = smooth_frequency(self.X_test, sigma=self.smoother_width)
            logging.info("Smoothing the frequency domain of the data.")
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            logging.info(f"New time delocalization: {X_loc.mean().item():.3g}")
            logging.info(
                f"New frequency delocalization: {X_spec_loc.mean().item():.3g}"
            )

    def download_data(self) -> None:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "shayanfazeli/heartbeat", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "ecg"


class SyntheticDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        max_len: int = 100,
        num_samples: int = 1000,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.max_len = max_len
        self.num_samples = num_samples

    def setup(self, stage: str = "fit") -> None:
        # Read CSV; extract features and labels
        path_train = self.data_dir / "train.csv"
        path_test = self.data_dir / "test.csv"

        # Read data
        df_train = pd.read_csv(path_train, header=None)
        X_train = df_train.values

        df_test = pd.read_csv(path_test, header=None)
        X_test = df_test.values

        # Convert to tensor
        self.X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
            2
        )  # Add a channel dimension
        self.y_train = None
        self.X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
        self.y_test = None

    def download_data(self) -> None:
        # Generate data, same DGP as in Fourier flows

        n_generated = 2 * self.num_samples  # For train + test
        phase = np.random.normal(size=(n_generated)).reshape(-1, 1)
        frequency = np.random.beta(a=2, b=2, size=(n_generated)).reshape(-1, 1)
        timesteps = np.arange(self.max_len)
        X = np.sin(timesteps * frequency + phase)
        X_train = X[: self.num_samples]
        X_test = X[self.num_samples :]

        # Save data
        df_train = pd.DataFrame(X_train)
        df_test = pd.DataFrame(X_test)
        df_train.to_csv(self.data_dir / "train.csv", index=False, header=False)
        df_test.to_csv(self.data_dir / "test.csv", index=False, header=False)

    @property
    def dataset_name(self) -> str:
        return "synthetic"


class MIMICIIIDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        n_feats: int = 40,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.n_feats = n_feats

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            mimic_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        # Load preprocessed tensors
        self.X_train = torch.load(self.data_dir / "X_train.pt")
        self.X_test = torch.load(self.data_dir / "X_test.pt")

        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)

        # Filter the tensors to keep the features with highest variance accross the population
        # The variance for each feature is averaged accrossed all time steps
        top_feats = torch.argsort(self.X_train.std(0).mean(0), descending=True)[
            : self.n_feats
        ]
        self.X_train = self.X_train[:, :, top_feats]
        self.X_test = self.X_test[:, :, top_feats]

    def download_data(self) -> None:
        dataset_path = self.data_dir / "all_hourly_data.h5"
        assert dataset_path.exists(), (
            f"Dataset {dataset_path} does not exist. "
            "Because MIMIC-III is a restricted dataset, you need to download it yourself. "
            "Our implementation relies on the default MIMIC-Extract preprocessed version of the dataset. "
            "Please follow the instruction on https://github.com/MLforHealth/MIMIC_Extract/tree/master."
        )

    @property
    def dataset_name(self) -> str:
        return "mimiciii"


class NASDAQDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            nasdaq_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        # Load preprocessed tensors
        self.X_train = torch.load(self.data_dir / "X_train.pt")
        self.X_test = torch.load(self.data_dir / "X_test.pt")

        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)
        assert self.X_train.shape[1:] == self.X_test.shape[1:] == (252, 6)

        # Filter out the last feature (volume) due to awkward scaling
        self.X_train = self.X_train[:, :, :-1]
        self.X_test = self.X_test[:, :, :-1]

    def download_data(self) -> None:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "jacksoncrow/stock-market-dataset", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "nasdaq"


class NASADatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        subdataset: str = "charge",
        remove_outlier_feature: bool = True,
    ) -> None:
        self.subdataset = subdataset
        self.remove_outlier_feature = remove_outlier_feature

        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / self.subdataset / "X_train.pt").exists()
            or not (self.data_dir / self.subdataset / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name}_{self.subdataset} not found. "
                f"Now running the preprocessing pipeline."
            )
            nasa_preprocess(
                data_dir=self.data_dir,
                subdataset=self.subdataset,
                random_seed=self.random_seed,
            )
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        # Load preprocessed tensors
        self.X_train = torch.load(self.data_dir / self.subdataset / "X_train.pt")
        self.X_test = torch.load(self.data_dir / self.subdataset / "X_test.pt")

        if self.remove_outlier_feature and self.subdataset == "charge":
            # Remove the third feature which has a bad range
            self.X_train = self.X_train[:, ::2, [0, 1, 3, 4]]
            self.X_test = self.X_test[:, ::2, [0, 1, 3, 4]]

            assert self.X_train.shape[2] == self.X_test.shape[2] == 4
            assert self.X_train.shape[1] == 251
            assert self.X_test.shape[1] == 251
        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)

    def download_data(self) -> None:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "patrickfleith/nasa-battery-dataset", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "nasa"


class USDroughtsDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            droughts_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)

            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        # Load preprocessed tensors
        self.X_train: torch.Tensor = torch.load(self.data_dir / "X_train.pt")
        self.X_test: torch.Tensor = torch.load(self.data_dir / "X_test.pt")

        # Remove features that have high correlation with T2M
        feats = [i for i in range(self.X_train.shape[2]) if i not in {4, 5, 6, 7, 9}]
        self.X_train = self.X_train[:, :, feats]
        self.X_test = self.X_test[:, :, feats]

        # Check tensors
        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)
        assert self.X_train.shape[1] % 365 == self.X_test.shape[1] % 365 == 0
        assert self.X_train.shape[2] == self.X_test.shape[2] == len(feats)

    def download_data(self) -> None:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "cdminix/us-drought-meteorological-data", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "droughts"
