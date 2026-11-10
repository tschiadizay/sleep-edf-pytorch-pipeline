"""Create stratified grouped K-Fold CV for Sleep-EDF[1, 2] in pytorch.

Assumes:
    convert_data_to_hdf5.py and sort_and_filter_data.py have been ran.

Results:
    a torch.utils.data.Dataset for SleepEDF where each item
    has the following format:
        x = (n_channels, 3000) (one epoch with n_channels at 3000 samples)
        y = (,) (one label between 0 and 7 with the following encoding)
    has a method for creating a stratified grouped K-Fold Cross Validation set.

Sources:
    [1] B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a
    sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
    EEG. IEEE-BME 47(9):1185-1194 (2000). https://ieeexplore.ieee.org/document/867928/

    [2] Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C.,
    Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform
    for biomedical research. Nature Health.
    https://doi.org/10.1038/s44360-026-00096-z. Available from:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC13641116/
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, cast

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Config class for creating a grouped stratified K-Fold.

    Args:
        csv_path (str | Path): path of the
            Processed_Sleep-{study-type}-DATA.csv.
        study_type (Literal["cassette", "telemetry"]): desired study_type.
        stratify_over_attributes (list[str]): list of attributes as
            attr_keys which should be stratified over.
        n_splits (int): the number of splits/the k in K-fold cross
            validation.
        shuffle (bool): Whether the data should be shuffled before folding.
        random_state (int | None, optional): seed for randomness.
            Defaults to None.

    """

    csv_path : str | Path
    study_type : Literal["telemetry", "cassette"]
    stratify_over_attributes : list[str]
    n_splits : int
    shuffle : bool
    random_state : int | None = None

class SleepEDFDataset(torch.utils.data.Dataset):
    """Implement torch.utils.data.Dataset for .hdf5 files for SleepEDF.

    Basic torch.utils.data.Dataset for SLEEPEDF from Physionet with Stratified
    grouped K-Fold Capability and more, i.e:
        gives stratified grouped K-Fold Cross Validation splits over custom
            attributes.
        gives attribute domains.

    Object attributes:
        self.hdf5_path (str | Path | None, optional): path to the hdf5_file
            for __getitem__. Not needed for other functions.
        self.h5py_file (h5py.File | None): h5py.File file object or None,
            used to lazily open h5py file for multiple workers.
        self.filter_sessions (list[str]): list of session_keys which should be
            included in the dataset.
        self.limits (torch.Tensor): tensor of length,
            len(self.filter_sessions)+1, where for each session i, at index i
            the start epoch and at index i+1 the end epoch (exclusive) is given,
            if all epochs would be inserted in a 1d list in order of
            filter_sessions. Used in __getitem__ to get the asked
            epoch.
        self.attrs (dict[str, list]): contains for each attr_key a list of
            attribute values in order of the self.filter_sessions.
    """

    all_cassette: ClassVar[set[str]] = {
        f"{i:02d}-{j}" for i in range(83) for j in range(1, 3)}
    all_cassette -= {"13-2","36-1","39-1","39-2", "52-1","68-1","68-2",
                     "69-1","69-2","78-1","78-2","79-1","79-2"}

    all_telemetry: ClassVar[set[str]] = {
        f"{i:02d}-{j}" for i in range(25) for j in range(1, 3)}
    all_telemetry -= {"00-1", "00-2", "23-1", "23-2", "03-1", "03-2"}

    all_attrs_cassette : ClassVar[list[str]] = [
        "age", "filename_hypnogram", "filename_psg", "lights_off_time",
        "measurement_date", "n_epochs", "night", "random_seventh_letter",
        "scorer", "sex", "subject"]

    all_attrs_telemetry : ClassVar[list[str]]= [
        *all_attrs_cassette, "is_placebo_session"]

    valid_keys : ClassVar[dict[str, tuple[list[str], list[str]]]]= {
        "cassette" : (sorted(all_cassette), all_attrs_cassette),
        "telemetry" : (sorted(all_telemetry), all_attrs_telemetry)}

    def __init__(
            self,
            hdf5_path : str | Path | None, csv_path : str | Path,
            study_type : Literal["cassette", "telemetry"],
            filter_sessions : list[str], filter_attrs : list[str]):
        """Initialize torch.utils.data.Dataset object.

        Args:
            hdf5_path (str | Path | None): path to the hdf5_file
            for __getitem__. Not needed for other functions.
            csv_path (str | Path): path of the
                Processed_Sleep-{study-type}-DATA.csv.
            study_type (Literal["cassette", "telemetry"]): desired study_type.
            filter_sessions (list[str]): Session ids which should be
                inside resulting dataset.
            filter_attrs (list[str]): attribute_names which should be
                inside resulting dataset.

        """
        assert study_type in SleepEDFDataset.valid_keys
        assert set(filter_sessions) <= set(
            SleepEDFDataset.valid_keys[study_type][0])
        assert set(filter_attrs) <= (
            set(pd.read_csv(csv_path, nrows=0, dtype=str).columns)
        )

        self.hdf5_path = hdf5_path
        self.h5py_file = None # only lazy_read the h5py file

        self.filter_sessions = filter_sessions

        self._create_index_and_read_metadata(csv_path, filter_attrs)

    def _create_index_and_read_metadata(self, csv_path, filter_attrs):
        df = pd.read_csv(csv_path, index_col="INDEX", dtype=str)
        # only read the columns of interesting sessions
        df = df.loc[self.filter_sessions, :]

        # get n_epochs of sessions
        n_epochs = torch.tensor(
            df.loc[:, "n_epochs"].astype("int").values, dtype=torch.int32)
        # Define self.limits which stores the inclusive beginning and exclusive
        # ending of the i.th session in self.limits[i] and self.limits[i+1]
        self.limits = torch.nn.functional.pad(
            torch.cumsum(n_epochs, dim=0),  pad=(1, 0), value=0)

        # store for each attribute_key from filter_attributes the read values
        self.attrs = {attr_key : df.loc[:, attr_key].tolist()
                      for attr_key in [*filter_attrs, "subject"]}

    def __len__(self) -> int:
        """Return length of the dataset in epochs.

        Returns:
            (int): length of the dataset in number of epochs.

        """
        return int(self.limits[-1].item())

    def _lazy_open(self):
        """Open lazily h5py_file for multiple node workers."""
        if self.h5py_file is None:
            assert self.hdf5_path is not None
            self.h5py_file = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one epoch-long recording with label.

        Args:
            idx (int): Index of the epoch in the transformed dataset.

        Returns:
            torch.Tensor : PSG recording.
                Shape: (n_channels, epoch_duration * sample_frequency)
                Dtype: float32
            torch.Tensor : Hypnogram class labeled by scorer.
                Shape : (,)
                Dtype: long : Hypnogram class with shape (,) labeled by the

        """
        self._lazy_open()
        session_key_start_index = torch.searchsorted(
            self.limits[1::], idx, side="right")
        session_key = self.filter_sessions[session_key_start_index]
        epoch_key =  idx - self.limits[session_key_start_index]
        assert self.h5py_file is not None
        dataset = cast(dict, self.h5py_file[session_key])
        recording = torch.from_numpy(dataset["psg"][epoch_key]).float()
        label = torch.tensor(dataset["hypnogram"][epoch_key]).long()
        return (recording, label)

    def create_encoded_class_strings(
            self, class_attributes : list[str]) -> list[str]:
        """Encode session to class_membership.

        Because one might stratify over multiple attributes (for example age and
        scorer), combine the subclasses into one class by concatenating their
        string representation. As the SLEEPEDF is really small, the only useful
        stratify option is over age into 4 classes.

        Args:
            class_attributes (list[str]): list of attr_keys
                which should be combined into one class.

        Returns:
            list[str]: Strings for each session displaying the combined class,
                i.e. "0-M-A" for a Male Guy under 25 years old scored by the
                scorer "A".

        """
        assert set(class_attributes) <= self.attrs.keys()
        # store for each session_key a string which encodes the attributes
        n_sessions = len(self.filter_sessions)
        encoded_class_list = [[] for _ in range(n_sessions)]
        if "age" in class_attributes:
            n_buckets = 4
            ages = np.array(self.attrs["age"], dtype=int)
            sorted_ages =  np.sort(ages)
            buckets = np.array_split(sorted_ages, n_buckets)
            limits = [i[-1] for i in buckets]
            for i in range(n_sessions):
                encoded_class_list[i].append(
                    str(np.searchsorted(limits, ages[i], side="left").item()))
        for attr_key in class_attributes:
            if attr_key == "age":
                # already processed
                continue
            for i in range(n_sessions):
                encoded_class_list[i].append(self.attrs[attr_key][i])
        encoded_class_list = ["-".join(item) for item in encoded_class_list]
        return encoded_class_list

    @staticmethod
    def create_stratified_split_of_subjects(
        config : Config) -> list[tuple[list[str], list[str]]]:
        """Create a stratified and grouped Cross_Validation_Split.

        Creates a stratified over stratify_over_attributes and grouped by
        subject_names n_splits-Cross-Validation. If we group by subject,
        then we automatically group by night_session which is also of course
        necessary. (One subject might have multiple night recordings,
        but one night recording only has one subject)

        Args:
            config (Config): filled out Config object.

        Returns:
            list[tuple[list[str], list[str]]]: a list of length n_splits. Each
                item is a tuple consisting of session_keys for the training_set
                and of session_keys for the test_set.

        """
        # Create a torch.utils.data.Dataset of the entire SleepEDFDataset
        # study_type to fetch attributes of the sessions and the resulting
        # classes
        sleepedf_obj = SleepEDFDataset(
            None, config.csv_path, config.study_type,
            SleepEDFDataset.valid_keys[config.study_type][0],
            SleepEDFDataset.valid_keys[config.study_type][1])
        encoded_class_list = sleepedf_obj.create_encoded_class_strings(
            config.stratify_over_attributes)
        # Stratify over encoded_class_list and group by subject_names for each
        # session
        skf_obj = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=config.shuffle,
            random_state=config.random_state)
        splits_as_session_indices = skf_obj.split(
            np.zeros(len(encoded_class_list)),
            encoded_class_list, groups=sleepedf_obj.attrs["subject"])
        # Store the splits not as indices of subject indices, but as
        # session_keys
        splits = []
        for train_indices, test_indices in splits_as_session_indices:
            train_sessions = [sleepedf_obj.filter_sessions[session_index]
                              for session_index in train_indices]
            test_sessions = [sleepedf_obj.filter_sessions[session_index]
                             for session_index in test_indices]
            splits.append((train_sessions, test_sessions))
        return splits

if __name__ == "__main__":
    import utils.logging_setup
    utils.logging_setup.set_up_logging()
    logger = logging.getLogger(__name__)

    logger.info("Started %s", __name__)

    # Example Code to create a stratified grouped K-Fold split of the SleepEDF
    # Dataset and how one train_loader/test_loader might be created.
    base_dir = Path(
        "") # TODO
    base_dir = base_dir / "SleepEDFConverted"
    study_types : list[Literal["cassette", "telemetry"]] = [
        "cassette", "telemetry"]

    for test_study_type in study_types:
        logger.info("Loading %s as a dataset started:", test_study_type)
        test_hdf5_path = base_dir / f"Processed_Sleep-{test_study_type}.hdf5"
        test_csv_path = base_dir / f"Processed_Sleep-{test_study_type}-DATA.csv"
        # splits is a list of length n_splits and contains a tuple of
        # training and test session ids
        test_splits = SleepEDFDataset.create_stratified_split_of_subjects(
            Config(test_csv_path, test_study_type, ["age"], 5, True, None))

        logger.info("Stratifiyng Split completed for %s", test_study_type)

        # Load entire dataset into memory
        all_session_names = (SleepEDFDataset.all_cassette if test_study_type ==
                             "cassette" else SleepEDFDataset.all_telemetry)
        test_dataset = SleepEDFDataset(
            test_hdf5_path, test_csv_path,test_study_type,
            SleepEDFDataset.valid_keys[test_study_type][0],
            SleepEDFDataset.valid_keys[test_study_type][1])

        for fold_idx, (train_idx_set, val_idx_set) in enumerate(test_splits):
            logger.debug(
                "fold_idx=%d completed for test_study_type=%s", fold_idx,
                test_study_type)
            train_data = SleepEDFDataset(
                test_hdf5_path, test_csv_path,test_study_type, train_idx_set,
                SleepEDFDataset.valid_keys[test_study_type][1])
            val_data = SleepEDFDataset(
                test_hdf5_path, test_csv_path, test_study_type, val_idx_set,
                SleepEDFDataset.valid_keys[test_study_type][1])
            # create a training set from stratified grouped data in splits
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=200, shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=200, shuffle=False, num_workers=4)
