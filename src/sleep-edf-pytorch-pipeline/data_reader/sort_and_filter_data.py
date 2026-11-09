"""Convert Sleep-EDF[1,2] dataset to .hdf5 with different data treatments.

After running convert_data_to_hdf5.py, the psg recordings and annotions have not
been splitted, filtered or sorted yet. This script reads the .hdf5 files and
creates new .hdf5 files in which:
    Before, te annotions of the dataset are not given on a per epoch basis, but
        on a continuous stage basis, making data parsing more complicated.
        This is why we store the hdf5 files on a (sessions, epochs) basis
        aligned with annotations.
    Align PSG with Hypnogram because Hypnogram starts in a lot of cases later.
    Missing epoch annotations are added with the label "Sleep stage ?": 0.
    Channel_names channels can be filtered i.e. to to reduce file size.
    Epoch annotations can be subsumed under new annotitations (i.e. to convert
        R & K to AASM).
    Epoch can be filtered based on a set of unwanted annotations.
    Remove n_trim_epochs Wake-epochs with respect to first/last Non-Wake-Epoch.
    Store the processed metadata inside the old.csv, i.e. number of epochs,
        start_onset, number of filtered_epochs, etc.

Assumptions:
    Assumes that convert_data_to_hdf5.py has been successfully executed.

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
from typing import cast

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

ENCODING: dict[str, int] = {
    "Sleep stage ?": 0,
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 5,
    "Sleep stage R": 6,
    "Movement time": 7,
}
REVERSED_ENCODING: dict[int, str] = {
    value : key for (key, value) in ENCODING.items()}

# pylint: disable=too-many-instance-attributes
@dataclass
class Config:
    """Store function parameters needed for one entire study_type.

    Attributes:
        metadata_path (str | Path): path of the info object stored in .fif
            file, created by convert_data_to_hdf5.py.
        in_path (str | Path): path of the .hdf5 file, created by
            convert_data_to_hdf5.py.
        out_path (str | Path): desired folder for the results.
        original_csv_path (str | Path): path of the (old) .csv file,
            created by convert_data_to_hdf5.py.
        new_csv_path (str | Path): desired path for the new csv file.
        chunk_size (int): chunk_size of the resulting .hdf5 file.
        sample_frequency (int | None): used sample_frequency in recordings,
            is read by the given metadata automatically, therefore should be
            set to None initialliy.
        filter_channels (list[str]): channels which should be stored
            in resulting .hdf5  in the desired order.
        channel_filters (list[tuple[float | None, float | None]]):
            map each channel_name from filter_channels to a l_freq and h_freq
            frequency or None.
        filter_channel_indices (list[int]): filter_channels but mapped
            to the index of the channel_name from study_metadata["ch_names"]
            (is automatically filled out).
        epoch_duration (int): duration of an epoch in seconds.
        n_trim_epochs (int): number of epochs which should be removed
            before/after first/last NonWake Epoch.
        encoding_mapping (dict[int, int | None]): dict mapping sleep stage
            labels from 0 to 7, or None if the sleep stage should be removed.

    Warnings:
        encoding_mapping has to map values starting from 0 to
            _desired_n_classes-1.

    """

    metadata_path : str | Path
    in_path: str | Path
    out_path: str | Path
    original_csv_path : str | Path
    new_csv_path : str | Path
    chunk_size: int
    sample_frequency: int | None
    filter_channels: list[str]
    channel_filters: list[tuple[float | None, float | None]]
    filter_channel_indices: list[int] | None
    epoch_duration: int
    n_trim_epochs: int
    encoding_mapping: dict[int, int | None]

    def __post_init__(self):
        """Validate some errorprone fields."""
        if self.encoding_mapping:
            assert set(self.encoding_mapping.keys()) <= {0,1,2,3,4,5,6,7}
            assert set(self.encoding_mapping.values()) <= {0,1,2,3,4,5,6,7,None}
            values = set(self.encoding_mapping.values()) - {None}
            assert sorted(values) == list(range(len(values))) # type: ignore


@dataclass
class ProcessedSession:
    """Store parameters needed for the transformation of one session.

    Attributes:
        psg (torch.Tensor): processed psg_recording of
            Shape: `(n_epochs, n_channels, epoch_duration * sample_frequency)`,
            dtype: `torch.float32`
        hypno (torch.Tensor): processed hypnogram
            Shape: `(n_epochs,)`,
            dtype: `int`
        recording_onset (int): the onset of the first onset in the hypnogram.
        cut_begin (int): first epoch which is included after trimming.
        cut_end (int): cut_off_index after trimming (index itself is not
            included).
        filtered_epochs (dict[str, str]): Maps filtered_epoch_names to
            epoch_indices which were removed. To be stored in a .csv,
            epoch_indices are stored as a joined string.
        n_epochs (int): the final number of epochs stored in the recordings.
        n_sleep_stages (dict[int, int]): Maps number of sleep stages in final
            recording, for example to calculate class weights.

    """

    psg : torch.Tensor
    hypno : torch.Tensor
    recording_onset : int
    cut_begin : int
    cut_end : int
    filtered_epochs : dict[str, str]
    n_epochs: int
    n_sleep_stages : dict[int, int]

def get_metadata(config : Config):
    """Fill out filter_channel_indices and sample_frequency in config object.

    Args:
        config (Config) : Config object.

    Results:
        config.filter_channel_indices is filled out by the value(s) read in
            config.metadat_path.
        config.sample_frequency is filled out by the value(s) read in
            config.metadat_path.

    """
    # get indices of needed channels by reading the study metadata of the
    # .fif file in metadata_path
    study_metadata = mne.io.read_info(config.metadata_path, verbose="error")
    channel_names = study_metadata["ch_names"]
    channel_map = { ch_name : idx
                    for idx, ch_name in enumerate(channel_names)}
    # sort channel_idx because h5py needs increasing indices
    config.filter_channel_indices = [
        channel_map[i] for i in config.filter_channels]
    # get used sample frequency
    config.sample_frequency = int(study_metadata["sfreq"])

def get_channel_filtered_session_data(
        src : h5py.File, session_key : str, filter_channel_indices : list[int]
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get PSG and Hypno data of the given session_key and ordered channels.

    Args:
        src (h5py.File): Already open h5py file created by convert_data_to_hdf5.
        session_key (str): Session_key as string uniquely identifying session.
        filter_channel_indices (list[int]): Indices of desired channels in the
            desired order.

    Returns:
        psg (torch.Tensor): psg recording of original .hdf5 file.
            Shape: `(n_channels, n_samples)`, dtype: torch.float32
        hypno (torch.Tensor): hypnogram annotation of original .hdf5 file in
            where each value of the 3-dimensional list is onset in s,
            duration in s, and decription in 0 to 7 (see Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int

    """
    # get psg and hypno from src.hdf5 and filter channels
    sorted_filter_channel_indices = sorted(filter_channel_indices)
    psg = torch.from_numpy(
        src[session_key]["PSG_DATA"] # type: ignore
        [sorted_filter_channel_indices, :])
    restored_order = [sorted_filter_channel_indices.index(ch_idx)
                      for ch_idx in filter_channel_indices]
    psg = psg[restored_order, :]
    # hynp_data has shape: (3, n_continuous_sleep_phases) with
    # (onset [in s], duration [in s], sleep_stage [0-7])
    hypno = torch.from_numpy(src[session_key]["HYPNOGRAM_DATA"] # type: ignore
                                    [:]).int()
    return psg, hypno

def filter_signals(
        psg: torch.Tensor, sampling_rate : float,
        mapping_channel_to_bandpass :list[tuple[float | None, float | None]]
        ) -> torch.Tensor:
    """Filter signals in psg according to AASM guidelines.

    Args:
        psg (torch.Tensor): psg recording.
            Shape: `(n_channels, n_samples)`, dtype: torch.float32
        sampling_rate (float): samlping rate of all channels.
        mapping_channel_to_bandpass (list[tuple[float, float]]):
            (l_freq, h_freq) for each channel, else None.

    Returns:
        psg (torch.Tensor): filtered psg recording.
            Shape: `(n_channels, n_samples)`, dtype: torch.float32

    """
    filtered_psg = psg.detach().numpy().copy().astype(np.float64)
    for i, (l_freq, h_freq) in enumerate(mapping_channel_to_bandpass):
        filtered_psg[i] = mne.filter.filter_data(
            filtered_psg[i], sampling_rate, l_freq=l_freq, h_freq=h_freq,
            n_jobs=1, verbose="WARNING")
    return torch.from_numpy(filtered_psg)
def correct_errors_in_hypno(hypno: torch.Tensor
                               ) -> torch.Tensor:
    """Correct missing annotations by adding additional labels with ?.

    In SLEEPEDF dataset, in the telemetry study, for session "12-1" and "22-1",
    there are for two timewindows missing annotations which we add by labeling,
    them with "Sleep stage ?".

    Args:
        hypno (torch.Tensor): hypnogram annotation of original .hdf5 file in
            where each value of the 3-dimensional list is onset in s,
            duration in s, and decription in 0 to 7 (see Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int

    Returns:
        corrected_hypno (torch.Tensor): corrected hypnogram annotation of
            original .hdf5 file in where each value of the 3-dimensional list
            is onset in s, duration in s, and decription in 0 to 7 (see
            Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int

    """
    onsets = hypno[0, :]
    durations = hypno[1, :]

    # onsets[0] should be 0 but just to be sure
    time_onsets_from_durations = torch.cumsum(durations, 0) + onsets[0]
    if torch.equal(time_onsets_from_durations[:-1], onsets[1:]):
        # no padding is needed because everything is correct
        return hypno

    corrected_hypno = []
    for i in range(len(onsets)):
        corrected_hypno.append(
            [hypno[0, i], hypno[1, i], hypno[2, i]]
        )
        # if onset_i + duration_i != onset_{i+1}
        if (i < len(onsets) - 1) and (
            hypno[0, i] + hypno[1, i] != hypno[0, i+1]):
            new_onset = hypno[0, i] + hypno[1, i]
            new_duration = hypno[0, i+1] - new_onset
            new_label = ENCODING["Sleep stage ?"]
            corrected_hypno.append(
                [new_onset, new_duration, new_label]
            )
    return torch.tensor(corrected_hypno).T

def remove_onset_offset_from_hypno_and_psg(
        psg: torch.Tensor, hypno: torch.Tensor, sfreq : int
        ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Remove offset of onset from hypno so hypno and psg are aligned.

    The hypnogram annotations start sometimes later than the actual psg
    recording, as seen in the first onset of each hypnogram. We remove
    this offset from every annotation in the hypnogram and throw away
    the first first_onset*sfreq samples in the psg recording so that
    the psg recording and the hypnogram annotations are perfectly aligned.

    Args:
        psg (torch.Tensor): psg recording.
            Shape: `(n_channels, n_samples)`, dtype: torch.float32
        hypno (torch.Tensor): hypnogram annotation
            where each value of the 3-dimensional list is onset in s,
            duration in s, and decription in 0 to 7 (see Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int
        sfreq (int) : Sample Frequency in psg recording.

    Returns:
        psg (torch.Tensor): psg recording.
            Shape: `(n_channels, n_samples - onset[0])`, dtype: torch.float32
        hypno (torch.Tensor): hypnogram annotation
            where each value of the 3-dimensional list is onset in s,
            duration (-onset[0]) in s, and decription in 0 to 7
            (see Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int
        first_onset (int): the onset of the first annotation (the offset).

    """
    first_onset = int(hypno[0, 0])
    hypno[0] = hypno[0] - first_onset
    psg = psg[:, first_onset*sfreq::]
    return psg, hypno, first_onset

def get_aligned_psg_hypno(
        psg: torch.Tensor, hypno: torch.Tensor, epoch_duration: int,
        sample_frequency: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Align (and reshape) psg and hypno by shortening both to shorter duration.

    The psg recording and hypnogram annotations have, even after aligning,
    sometimes different durations. To fix this, we choose the shorter one and
    only take of both recordings the first n_epochs, where n_epochs is the
    duration of the shorter psg/hypnogram. Afterwards, we transform the
    hypnogram from (3, n_continuous_sleep_phases) to (n_epochs,), where
    each item encodes the label of an epoch_duration long epoch. We also
    transform the psg recording to
    (n_epochs, n_channels, epoch_duration * sample_frequency).

    Args:
        psg (torch.Tensor): psg recording.
            Shape: `(n_channels, n_samples - onset[0])`, dtype: torch.float32
        hypno (torch.Tensor): hypnogram annotation
            where each value of the 3-dimensional list is onset in s,
            duration (-onset[0]) in s, and decription in 0 to 7
            (see Encoding above.).
            Shape: `(3, n_continuous_sleep_phases)`, dtype: int
        epoch_duration (int) : Desired duration of an epoch. Should be 30.
        sample_frequency (int) : Sample Frequency in psg recording.

    Returns:
        psg (torch.Tensor): psg recording.
            Shape: `(n_epochs, n_channels, epoch_duration * sample_frequency)`,
            dtype: torch.float32
        hypno (torch.Tensor): list of labels of epoch_duration long epochs.
            Shape: `(n_epochs,)`, dtype: int

    """
    # find common n_epochs number from psg and hypno data
    duration_in_epochs_psg = psg.shape[1] // (
        epoch_duration * sample_frequency)
    # end_onset - start_onset + duration of last phase
    duration_hypno = int(hypno[0, -1] - hypno[0,0] + hypno[1, -1])
    # all annotations are done in timewindows of multiples of 30s
    # this is why this floor division works without a remainder
    duration_in_epochs_hypno =  duration_hypno // epoch_duration
    n_epochs = min(
        [duration_in_epochs_psg, duration_in_epochs_hypno])

    # get duration in s and labels of continuous sleep phases
    dur_of_contin_phases = hypno[1, :]
    labels_of_contin_phases = hypno[2, :]

    # repeat each label of one continuous sleep phase epoch-times,
    # but before convert dur_fo_contin_phases from s to epochs
    hypno = torch.repeat_interleave(
        labels_of_contin_phases, dur_of_contin_phases // epoch_duration)
    # take only the first n_epochs to align them both
    hypno = hypno[:n_epochs]
    psg = psg[
    :, :n_epochs * epoch_duration * sample_frequency]
    # convert psg from (n_channels, duration [in samples])
    # to (n_epochs, n_channels, epoch_duration * sample_frequency)
    n_channels = psg.shape[0]
    psg = psg.view(
        n_channels, n_epochs, epoch_duration * sample_frequency)
    psg = psg.permute(1, 0, 2)
    return psg, hypno

def trim_epochs_and_labels(psg : torch.Tensor,
                           hypno : torch.Tensor, n_trim_epochs : int
                           ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Trim epochs and labels n_trim_epochs (before first)/(after last) NonWake.

    Args:
        psg (torch.Tensor): psg recording.
            Shape: `(n_epochs, n_channels, epoch_duration * sample_frequency)`,
            dtype: torch.float32
        hypno (torch.Tensor): list of labels of epoch_duration long epochs.
            Shape: `(n_epochs,)`, dtype: int
        n_trim_epochs (int): Number of epochs which should be removed
        (before first)/(after last) not wake epoch.

    Returns:
        psg (torch.Tensor): trimmed psg recording.
            Shape: `(cut_begin:cut_end, n_channels,
            epoch_duration * sample_frequency)`,
            dtype: torch.float32
        hypno (torch.Tensor): trimmed list of labels of epoch_duration long
            epochs.
            Shape: `(cut_begin:cut_end,)`, dtype: int
        cut_begin: original_index of first included epoch
        cut_end: the smallest index of epochs not included in the
            trimmed psg/hypno. (the exclusive upper limit).

    Warnings:
        "Sleep stage ?": 0,"Sleep stage W": 1, "Movement time" are considered
        as Non Wake stage.

    """
    # Exclude "Sleep stage ?": 0,"Sleep stage W": 1, "Movement time":
    nwake_strings = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3',
                     'Sleep stage 4', 'Sleep stage R',]
    nwake_labels = torch.Tensor([ENCODING[key] for key in nwake_strings])
    nwake_mask = torch.isin(hypno, nwake_labels)

    first_nwake_epoch = int(torch.argmax(nwake_mask.int()))
    last_nwake_epoch = int(nwake_mask.shape[0] - 1 - torch.argmax(
        nwake_mask.int().flip(0)))

    cut_begin = max([first_nwake_epoch - n_trim_epochs, 0])
    cut_end = min([last_nwake_epoch + n_trim_epochs + 1, nwake_mask.shape[0]])

    psg = psg[cut_begin:cut_end, :, :]
    hypno = hypno[cut_begin:cut_end]

    return psg, hypno, cut_begin, cut_end

def filter_by_annotations(psg: torch.Tensor, hypno: torch.Tensor,
                          encoding_mapping: dict[int, int | None]
                          ) -> tuple[torch.Tensor, torch.Tensor,
                                     dict[str, str]]:
    """Filter psg and hypno data based on a set of forbidden stages.

    Args:
        psg (torch.Tensor): trimmed psg recording.
            Shape: `(cut_begin:cut_end, n_channels,
            epoch_duration * sample_frequency)`,
            dtype: torch.float32
        hypno (torch.Tensor): trimmed list of labels of epoch_duration long
            epochs.
            Shape: `(cut_begin:cut_end,)`, dtype: int
        encoding_mapping (dict[int, int | None]): dict mapping sleep stage
            labels to None when they should be removed and otherwise an int.

    Returns:
        psg (torch.Tensor): filtered psg recording.
            Shape: `([mask], n_channels, epoch_duration * sample_frequency)`,
            dtype: torch.float32
        hypno (torch.Tensor): filtered list of labels of epoch_duration long
            epochs.
            Shape: `([mask],)`, dtype: int
        filtered_epochs (dict[str, str]): maps the to be removed key to the
            removed epoch indices. (The epoch indices orient themselves
            to the current indices, so after removing the offsets
            onset[0] and cut_begin).

    Warnings:
        The removed epoch indices orient themselves the current indices, so
            after removing the offsets onset[0] and cut_begin).

    """
    forbidden_stages = [key for key, value in encoding_mapping.items()
                        if value is None]
    filtered_epochs = {
        REVERSED_ENCODING[key] : ";".join(
            [str(i) for i in torch.where(hypno == key)[0].tolist()])
        for key in forbidden_stages
    }
    mask = ~ torch.isin(hypno, torch.tensor(forbidden_stages))
    return psg[mask], hypno[mask], filtered_epochs

def map_by_annotations(hypno: torch.Tensor,
                       encoding_mapping: dict[int, int | None]
                       ) -> torch.Tensor:
    """Map hypnogram data based on an encoding mapping.

    Args:
        hypno (torch.Tensor): filtered list of labels of epoch_duration long
            epochs.
            Shape: `([mask],)`, dtype: int
        encoding_mapping (dict[int, int | None]): dict mapping sleep stage
            labels to different stages. Mappings to None are ignored.

    Returns:
        hypno (torch.Tensor): mapped list of labels of epoch_duration long
            epochs.
            Shape: `([mask],)`, dtype: int

    """
    old_hypno = hypno.clone()
    for key, value in encoding_mapping.items():
        if value is not None:
            hypno[old_hypno == key] = value
    return hypno

def process_session(src: h5py.File, session_key : str, config : Config
                    ) -> ProcessedSession:
    """Process session and store results in newly created ProcessedSession obj.

    Args:
        src (h5py.File): source .hdf5 file created by convert_data_to_hdf5.py
            which is already opened, i.e. by a context_manager
        session_key (str): session_key of the to be read session
        config (Config): filled out config object (fields which can be None
        should not be set!)

    Returns:
        processed_session (ProcessedSession): filled out ProcessedSession
            object for the session.

    """
    # make sure that config.sample_frequency has been read
    assert config.sample_frequency is not None
    config.sample_frequency = cast(int, config.sample_frequency)
    # make sure that config.filter_channel_indices has been filled
    assert config.filter_channel_indices is not None

    # Get psg and hypno from src.hdf5 and filter channels
    psg, hypno = get_channel_filtered_session_data(
        src, session_key, config.filter_channel_indices)

    # Filter psg signal channels
    psg = filter_signals(psg, config.sample_frequency, config.channel_filters)
    # Remove offset of the onset in hypnograms and psg
    psg, hypno, recording_onset = remove_onset_offset_from_hypno_and_psg(
        psg, hypno, config.sample_frequency)

    # Pad missing annotations with new annotations labeled with unkown ?
    hypno = correct_errors_in_hypno(hypno)

    # Align psg and hypno recording by taking the shorter one and transform
    # dimensions.
    psg, hypno = get_aligned_psg_hypno(
        psg, hypno, config.epoch_duration, config.sample_frequency)

    # Trim data n_trim_epochs before/after first/last NWake label
    psg, hypno, cut_begin, cut_end = trim_epochs_and_labels(
        psg, hypno, config.n_trim_epochs)

    # Filter sleep stages where mapping is None
    psg, hypno, filtered_epochs = filter_by_annotations(
        psg, hypno, config.encoding_mapping)

    # Map epochs to different labels
    hypno = map_by_annotations(hypno, config.encoding_mapping)

    # Get number of epochs of final psg/hypnogram
    n_epochs = len(hypno)

    # Store number of occurences in final recording for each sleep stage
    counts = torch.bincount(hypno)
    n_sleep_stages = {
        sleep_stage_key : int(count.item())
        for sleep_stage_key, count in enumerate(counts)
    }

    # Store the processed session and obtained values as a processed_session obj
    processed_session = ProcessedSession(
        psg, hypno, recording_onset, cut_begin, cut_end, filtered_epochs,
        n_epochs, n_sleep_stages
    )

    return processed_session

def store_session(dst : h5py.File, session_key : str,
                  processed_session : ProcessedSession, config : Config):
    """Store processed_session into the dst file with the given config.

    Args:
        dst (h5py.File): destination hdf5 file where the session shall be
            stored.
        session_key (str): session_key of the to be stored session.
        processed_session (ProcessedSession): filled out ProcessedSession obj.
        config (Config): filled out config.

    Results:
        In the dst .hdf5 file, the processed psg recording can be opened under:
            "/<session_key>/<psg>"
        In the dst .hdf5 file, the processed hypnogram recording can be opened
            under:
            "/<session_key>/<hypnogram>"

    """
    # Create for the session_key a new dataset
    grp = dst.create_group(session_key)

    # Calculate chunk_size if resulting recording gets too small
    psg = processed_session.psg
    aligned_chunk_size = min(config.chunk_size, processed_session.n_epochs)
    chunk_shape = (
        min(config.chunk_size, processed_session.n_epochs), psg.shape[1],
        psg.shape[2])

    # Fill the dataset with psg data
    grp.create_dataset("psg", data=psg,
                        compression="gzip", shuffle=True,
                        chunks=chunk_shape)

    # Fill the dataset with hypnogram data
    hypno = processed_session.hypno
    grp.create_dataset("hypnogram", data=hypno,
                        compression="gzip", shuffle=True,
                        chunks=(aligned_chunk_size,))

def main(config : Config):
    """Do main procedure of sort_and_filter.

    Args:
        config (Config): Contains the parameters needed for the transformation
            of an entire study.

    """
    # get metadata of study
    get_metadata(config)

    in_path = config.in_path
    out_path = config.out_path
    metadata = []

    with h5py.File(in_path, "r") as src, h5py.File(out_path, "w") as dst:
        for session_key in tqdm(src, unit="session"):
            # for smooth type_annotation
            session_key = str(session_key)

            processed_session = process_session(src, session_key, config)

            subject, night = session_key.split("-")

            additional_metadata : dict = {
                "subject" : subject,
                "night" : night
            }
            additional_metadata.update(processed_session.n_sleep_stages)
            additional_metadata.update(processed_session.filtered_epochs)
            additional_metadata.update({
                "n_epochs" : processed_session.n_epochs,
                "recording_onsets" : processed_session.recording_onset,
                "cut_begins" : processed_session.cut_begin,
                "cut_ends" : processed_session.cut_end
            })
            metadata.append(additional_metadata)

            store_session(dst, session_key, processed_session, config)

        dst["/"].attrs["filter_channels"] = ";".join(config.filter_channels)
        dst["/"].attrs["sfreq"] = config.sample_frequency

    # Merge the metadata obtained by the sorting/filtering with the original.csv
    # to a new csv file.
    metadata = pd.DataFrame.from_records(metadata)
    metadata.index = metadata["subject"] + "-" + metadata["night"]
    original_df = pd.read_csv(config.original_csv_path, index_col=0, dtype=str)
    new_df = pd.merge(original_df, metadata, on=["subject", "night"])
    new_df.index = new_df["subject"] + "-" + new_df["night"]
    new_df.index.name = "INDEX"
    new_df.to_csv(config.new_csv_path, index=True)

if __name__== "__main__":
    import utils.logging_setup
    utils.logging_setup.set_up_logging()
    logger = logging.getLogger(__name__)

    logger.info("Started %s", __name__)

    # Preprocess the telemetry and cassette dataset with the above pipeline.
    # for details, see the main function and the corrosponding sub-pipeline
    # functions.
    base_dir = Path(
        "") # TODO
    for study_type in ["telemetry", "cassette"]:
        # Create Config
        main(Config(
            metadata_path = base_dir / f"{study_type}-metadata.fif",
            in_path = base_dir / f"Sleep-{study_type}.hdf5",
            out_path = base_dir / f"Processed_Sleep-{study_type}.hdf5",
            original_csv_path = base_dir / f"{study_type}-DATA.csv",
            new_csv_path = base_dir / f"Processed_Sleep-{study_type}-DATA.csv",
            chunk_size = 5,
            sample_frequency = None,
            filter_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal",
                               "EMG submental"],
            channel_filters= [(0.3, 35), (0.3, 35), (0.3, 35), (10, None)],
            filter_channel_indices = None,
            epoch_duration = 30,
            n_trim_epochs = 5,
            # Remove "Sleep stage ?" and "Movement time"
            # and map "Sleep stage 3" and "Sleep stage 4" to N3
            encoding_mapping = {
                0: None,
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 3,
                6: 4,
                7: None
            }
        ))
