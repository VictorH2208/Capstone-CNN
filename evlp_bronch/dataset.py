"""The BreathLabel type, dataset types, and data processing functions."""

import logging
import os
import re
from collections.abc import Iterable
from enum import Enum
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .util import get_project_root_dir

# # Configure the logging module
# logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# # logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# # Create a logger
_logger = logging.getLogger(__name__)

# # Create a console handler and set the level to DEBUG
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# # Create a formatter and set it for the console handler
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)

# # Add the console handler to the logger
# _logger.addHandler(console_handler)


# All lung IDs
ALL_LUNG_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    579,
    595,
    598,
    600,
    603,
    606,
    607,
    610,
    615,
    616,
    617,
    618,
    619,
    682,
    698,
    731,
    735,
    738,
    753,
    762,
    782,
    803,
    817,
    818,
)


class BreathLabel(Enum):
    """An enumeration that maps breath labels to integers."""

    # pylint: disable=invalid-name
    Normal = 0
    Assessment = 1
    Bronch = 2
    Recruitment = 3
    Noise = 4
    Unvisited = 5
    InPause = 6
    Deflation = 7
    Question = 8


class RawEVLPDataset(Dataset):
    """An EVLP dataset that are directly read from a directory of raw CSV files with
    minimal processing."""

    class Element(TypedDict):
        """An output element of `RawEVLPDataset`."""

        Lung_id: int
        Breath_num: np.ndarray
        P_peak: np.ndarray
        Dy_comp: np.ndarray
        Label: np.ndarray
        Max_gap: Optional[np.ndarray]
        In_vol: Optional[np.ndarray]
        Ex_vol: Optional[np.ndarray]
        IE_vol_ratio: Optional[np.ndarray]
        Duration: Optional[np.ndarray]
        In_duration: Optional[np.ndarray]
        Ex_duration: Optional[np.ndarray]
        IE_duration_ratio: Optional[np.ndarray]
        PEEP: Optional[np.ndarray]

    def __init__(
        self,
        lung_ids: Iterable[int],
        labeled_data_dir: Optional[str | os.PathLike] = None,
    ) -> None:
        _logger.debug(
            "Creating RawEVLPDataset of lung_ids=%s, labeled_data_dir='%s'",
            lung_ids,
            labeled_data_dir,
        )
        if labeled_data_dir is None:
            # We use "labeled" as opposed to "labelled" in code
            labeled_data_dir = get_project_root_dir() / "data/labelled_data"
        assert labeled_data_dir.is_dir(), f"'{labeled_data_dir}' is not a directory"
        filename_pattern = re.compile(r"(?:\d+_)?EVLP(?P<lung_id>\d+)_labeled.csv")
        unvisited_lung_ids = set(lung_ids)
        lung_id_to_file_path: dict[int, os.PathLike] = {}
        for file_path in labeled_data_dir.iterdir():
            match = filename_pattern.fullmatch(file_path.name)
            if match is None:
                continue
            lung_id = int(match.group("lung_id"))
            if lung_id in lung_id_to_file_path:
                _logger.warning("Multiple CSV files for lung_id %d", lung_id)
            elif lung_id in unvisited_lung_ids:
                unvisited_lung_ids.remove(lung_id)
                lung_id_to_file_path[lung_id] = file_path
        if unvisited_lung_ids:
            _logger.warning("No CSV files for lung_id %s", sorted(unvisited_lung_ids))
        self._elements = [self._read_csv_file(lung_id, file_path) for lung_id, file_path in sorted(lung_id_to_file_path.items())]

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, index: int) -> Element:
        return self._elements[index]

    def _read_csv_file(self, lung_id: int, file_path: os.PathLike) -> Element:
        _logger.debug("Reading lung_id=%d, file_path='%s'", lung_id, file_path)
        df = pd.read_csv(file_path)
        return {
            "Lung_id": lung_id,
            # Required fields
            **{column_name: df[column_name].to_numpy() for column_name in ("Breath_num", "P_peak", "Dy_comp")},
            "Label": pd.Categorical(df["Label"], [label.name for label in BreathLabel], ordered=True).codes,
            # Optional fields
            **{
                stripped_column_name: (df[raw_column_name].to_numpy() if raw_column_name in df else None)
                # Parentheses must be stripped off because of invalid variable names
                for raw_column_name, stripped_column_name in {
                    "Max_gap(ms)": "Max_gap",
                    "In_vol(ml)": "In_vol",
                    "Ex_vol(ml)": "Ex_vol",
                    "IE_vol_ratio": "IE_vol_ratio",
                    "Duration(s)": "Duration",
                    "In_duration(s)": "In_duration",
                    "Ex_duration(s)": "Ex_duration",
                    "IE_duration_ratio": "IE_duration_ratio",
                    "PEEP": "PEEP",
                }.items()
            },
        }


class ProcessedEVLPDataset(Dataset):
    """An EVLP dataset whose data have been processed and are ready for training. One
    CSV file may be mapped to multiple output elements of this dataset."""

    class Element(TypedDict):
        """An output element of `ProcessedEVLPDataset`."""

        # The following fields are copied from `RawEVLPDataset.Element`
        Lung_id: int
        Breath_num: np.ndarray
        P_peak: np.ndarray
        Dy_comp: np.ndarray
        Label: np.ndarray
        Max_gap: Optional[np.ndarray]
        In_vol: Optional[np.ndarray]
        Ex_vol: Optional[np.ndarray]
        IE_vol_ratio: Optional[np.ndarray]
        Duration: Optional[np.ndarray]
        In_duration: Optional[np.ndarray]
        Ex_duration: Optional[np.ndarray]
        IE_duration_ratio: Optional[np.ndarray]
        PEEP: Optional[np.ndarray]

        # One-hot labels used for model inputs
        Is_normal: np.ndarray
        Is_assessment: np.ndarray
        Is_bronch: np.ndarray

    def __init__(
        self,
        lung_ids: Iterable[int],
        labeled_data_dir: Optional[str] = None,
    ):
        _logger.debug(
            "Creating ProcessedEVLPDataset of lung_ids=%s, labeled_data_dir='%s'",
            lung_ids,
            labeled_data_dir,
        )
        self._raw_dataset = RawEVLPDataset(lung_ids, labeled_data_dir)
        self._processed_elements = []
        for raw_element in self._raw_dataset:
            self._processed_elements.extend(process_raw_dataset_element(raw_element))

    def __len__(self) -> int:
        return len(self._processed_elements)

    def __getitem__(self, index: int) -> Element:
        return self._processed_elements[index]


def process_raw_dataset_element(
    raw_element: RawEVLPDataset.Element,
) -> ProcessedEVLPDataset.Element:
    """
    1. Remove points
       - before the first normal / assessment label
       - after the last normal / assessment label

    2. Linearly interpolate points and add one-hot labels
       - between two normal labels or
       - between two assessment labels
       if there is no bronch in between

    3. Remove points of other labels
    4. Segment the series into multiple elements
    """
    indexes_raw_normal = np.nonzero(raw_element["Label"] == BreathLabel.Normal.value)[0]
    indexes_raw_assessment = np.nonzero(raw_element["Label"] == BreathLabel.Assessment.value)[0]

    # Remove points
    # before the first normal / assessment label
    # after the last normal / assessment label

    index_min = min(indexes_raw_normal[0], indexes_raw_assessment[0])
    index_max = max(indexes_raw_normal[-1], indexes_raw_assessment[-1])

    processed_element = {key: raw_element[key][index_min : index_max + 1] if raw_element[key] is not None and isinstance(raw_element[key], np.ndarray) else raw_element[key] for key in raw_element.keys()}

    processed_element.update(
        {
            "Is_normal": np.zeros_like(processed_element["Label"]),
            "Is_assessment": np.zeros_like(processed_element["Label"]),
            "Is_bronch": np.zeros_like(processed_element["Label"]),
        }
    )

    indexes_normal = np.nonzero(processed_element["Label"] == BreathLabel.Normal.value)[0]
    indexes_assessment = np.nonzero(processed_element["Label"] == BreathLabel.Assessment.value)[0]
    indexes_bronch = np.nonzero(processed_element["Label"] == BreathLabel.Bronch.value)[0]

    # Linearly interpolate points
    # between two normal labels or
    # between two assessment labels
    # there is no bronch in between

    for label in [BreathLabel.Normal, BreathLabel.Assessment]:
        if label == BreathLabel.Normal:
            indexes = indexes_normal
            is_label = "Is_normal"
            skip_label = BreathLabel.Assessment
            _logger.debug("Lung_id %d: Interpolation: Normal", processed_element["Lung_id"])
        else:
            indexes = indexes_assessment
            is_label = "Is_assessment"
            skip_label = BreathLabel.Normal
            _logger.debug("Lung_id %d: Interpolation: Assessment", processed_element["Lung_id"])
        for i, index in enumerate(indexes):
            processed_element[is_label][index] = 1
            if i == len(indexes) - 1:
                continue
            label_index_next = indexes[i + 1]
            # skip consecutive labels
            if index == label_index_next - 1:
                continue
            # if there is an bronch or assessment/normal label, delete the points between them
            if (processed_element["Label"][index + 1 : label_index_next] == skip_label.value).any() or (processed_element["Label"][index + 1 : label_index_next] == BreathLabel.Bronch.value).any():
                _logger.debug("Lung_id %d: Interpolation: delete starting from %d", processed_element["Lung_id"], index)
                continue

            for key in raw_element.keys():
                # linear interpolate
                if raw_element[key] is not None and isinstance(raw_element[key], np.ndarray) and key != "Label":
                    start = raw_element[key][index]
                    end = raw_element[key][label_index_next]
                    _logger.debug("Lung_id %d: Interpolation: between %d: %f and %d: %f", processed_element["Lung_id"], index, start, label_index_next, end)
                    diff = (end - start) / (label_index_next - index)
                    for j in range(index + 1, label_index_next):
                        processed_element[key][j] = processed_element[key][j - 1] + diff
                        processed_element[is_label][j] = 1

    # Remove points of other labels
    for key in raw_element.keys():
        if raw_element[key] is not None and isinstance(raw_element[key], np.ndarray) and key != "Label":
            processed_element[key] = processed_element[key].astype(float)
            processed_element[key][np.logical_and(processed_element["Is_normal"] != 1, processed_element["Is_assessment"] != 1)] = np.nan

    indexes_normal_index = 0
    indexes_assessment_index = 0
    # mark Is_bronch=1 for nearest normal/assessment label
    for _, index in enumerate(indexes_bronch):
        while index > indexes_normal[indexes_normal_index] and indexes_normal_index < len(indexes_normal) - 1:
            indexes_normal_index += 1
        while index > indexes_assessment[indexes_assessment_index] and indexes_assessment_index < len(indexes_assessment) - 1:
            indexes_assessment_index += 1
        if indexes_assessment_index != 0 and indexes_normal_index != 0:
            processed_element["Is_bronch"][max(indexes_normal[indexes_normal_index - 1], indexes_assessment[indexes_assessment_index - 1])] = 1
        elif indexes_assessment_index != 0:
            processed_element["Is_bronch"][indexes_assessment[indexes_assessment_index - 1]] = 1
        elif indexes_normal_index != 0:
            processed_element["Is_bronch"][indexes_normal[indexes_normal_index - 1]] = 1

    # Remove nan points
    processed_element["Is_normal"] = processed_element["Is_normal"][~np.isnan(processed_element["Dy_comp"])]
    processed_element["Is_assessment"] = processed_element["Is_assessment"][~np.isnan(processed_element["Dy_comp"])]
    processed_element["Is_bronch"] = processed_element["Is_bronch"][~np.isnan(processed_element["Dy_comp"])]
    for key in raw_element.keys():
        if raw_element[key] is not None and isinstance(raw_element[key], np.ndarray) and key != "Label":
            processed_element[key] = processed_element[key][~np.isnan(processed_element[key])]

    processed_elements = []
    indexes_is_bronch = np.nonzero(processed_element["Is_bronch"] == 1)[0]
    for i, index in enumerate(indexes_is_bronch):
        # print(index)
        if i == len(indexes_is_bronch) - 1:
            label_index_next = None
        else:
            label_index_next = indexes_is_bronch[i + 1]
        processed_elements.append({key: processed_element[key][:label_index_next] if processed_element[key] is not None and isinstance(processed_element[key], np.ndarray) else processed_element[key] for key in processed_element.keys()})
    # print(f"Lung_id {processed_element['Lung_id']}: {len(processed_elements)} elements")
    return processed_elements
