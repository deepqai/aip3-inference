"""
2018-2019 HTC Corporation. All Rights Reserved.
This source code is licensed under the HTC license which can be found in the
LICENSE file in the root directory of this work.
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from constants.preprocess import IMG_FOLDER
from utils import endecode


class BaseDataset(Dataset):
    def __init__(self, datadir, record_list, transform):
        self.path_prefix = os.path.join(datadir, IMG_FOLDER)
        self.record_list = record_list
        self.transform = transform

    def _get_input_data(self, file_path):
        if len(file_path) == 1:
            input_data = cv2.imread(
                os.path.join(self.path_prefix, file_path[0]), cv2.IMREAD_COLOR
            )
            return cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        input_data = []
        for path in file_path:
            input_data.append(
                cv2.imread(os.path.join(self.path_prefix, path), cv2.IMREAD_GRAYSCALE)
            )
        input_data = np.stack(input_data, 2)

        return input_data

    def __len__(self):
        return len(self.record_list)


class MultiLabelDataset(BaseDataset):
    """Describe and define a dataset format."""

    def __init__(self, datadir, record_list, transform):
        super().__init__(datadir, record_list, transform)

    def _get_label(self, record):
        return {"label": record["label"]}

    def __getitem__(self, idx):
        record = self.record_list[idx]
        file_path, filename = record["file_path"], record["original_name"]

        input_data = self._get_input_data(file_path)
        image = self.transform(image=input_data)["image"]

        file_path = (
            file_path[len(file_path) // 2] if len(file_path) > 1 else file_path[0]
        )
        data_dict = {"images": image, "file_paths": file_path, "filenames": filename}

        if record["label"]:
            label = self._get_label(record)
            label = torch.tensor(label["label"], dtype=torch.float)
            data_dict["targets"] = label

        return data_dict


class SingleLabelDataset(BaseDataset):
    """Describe and define a dataset format."""

    def __init__(self, datadir, record_list, transform):
        super().__init__(datadir, record_list, transform)

    def _get_label(self, record):
        return {"label": int(record["label"][0])}

    def __getitem__(self, idx):
        record = self.record_list[idx]
        file_path, filename = record["file_path"], record["original_name"]

        input_data = self._get_input_data(file_path)
        image = self.transform(image=input_data)["image"]

        file_path = (
            file_path[len(file_path) // 2] if len(file_path) > 1 else file_path[0]
        )
        data_dict = {"images": image, "file_paths": file_path, "filenames": filename}

        if record["label"]:
            label = self._get_label(record)
            label = torch.tensor(label["label"], dtype=torch.long)
            data_dict["targets"] = label

        return data_dict


class DetectionDataset(BaseDataset):
    def __init__(self, datadir, record_list, transform):
        super().__init__(datadir, record_list, transform)

    def _get_label(self, record):
        label = record["label"]

        bboxes = []
        labels = []

        for box in label:
            box = json.loads(box)
            if not box:
                continue

            lab, x1, y1, w, h, = (
                box["label"],
                box["bbox_x_min"],
                box["bbox_y_min"],
                box["bbox_width"],
                box["bbox_height"],
            )  # info extraction
            bboxes += [[x1, y1, w, h]]
            labels += [lab]

        return {"bboxes": bboxes, "labels": labels}

    def __getitem__(self, idx):
        record = self.record_list[idx]
        file_path, filename = record["file_path"], record["original_name"]

        input_data = self._get_input_data(file_path)

        file_path = (
            file_path[len(file_path) // 2] if len(file_path) > 1 else file_path[0]
        )

        data_need_transform = {"image": input_data}

        if record["label"]:
            label = self._get_label(record)
            data_need_transform["bboxes"] = label["bboxes"]
            data_need_transform["labels"] = label["labels"]

        # apply augmentation
        transformed = self.transform(**data_need_transform)

        # image value should be in [0, 1] due to torchvision detection model
        data_dict = {
            "images": transformed["image"] / 255,
            "file_paths": file_path,
            "filenames": filename,
        }

        if record["label"]:
            bboxes = torch.FloatTensor(transformed["bboxes"]).reshape(-1, 4)
            labels = torch.LongTensor(transformed["labels"])
            data_dict["targets"] = {"boxes": bboxes, "labels": labels}

        return data_dict


def od_collate_fn(batch):
    file_path_list, filename_list, images_list, target_list = [], [], [], []
    for b in batch:
        file_path_list += [b["file_paths"]]
        filename_list += [b["filenames"]]
        images_list += [b["images"]]

        if "targets" in b:
            target_list += [b["targets"]]

    data_dict = {
        "file_paths": file_path_list,
        "filenames": filename_list,
        "images": images_list,
    }

    if target_list:
        data_dict["targets"] = target_list

    return data_dict


class SegmentationDataset(BaseDataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth encoded mask).
    :param transform: transformations to apply.
    """

    def __init__(self, datadir, record_list, transform):
        super().__init__(datadir, record_list, transform)

    def _get_label(self, record):
        width, height = record["file_width"], record["file_height"]

        decoded_masks = []
        for mask in record["label"]:
            mask = endecode.decode(mask, width, height)
            decoded_masks.append(mask.astype(np.float32))

        return decoded_masks

    def __getitem__(self, idx):
        record = self.record_list[idx]
        file_path, filename = record["file_path"], record["original_name"]

        input_data = self._get_input_data(file_path)

        file_path = (
            file_path[len(file_path) // 2] if len(file_path) > 1 else file_path[0]
        )

        masks = None
        if record["label"]:
            masks = self._get_label(record)

        transformed = self.transform(image=input_data, masks=masks)

        data_dict = {
            "images": transformed["image"],
            "file_paths": file_path,
            "filenames": filename,
            "width": record["file_width"],
            "height": record["file_height"],
        }

        if masks:
            data_dict["targets"] = transformed["masks"]

        return data_dict
