import urllib
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou

from .endecode import encode
from .metric import Metric, numeric_score


class Logger:
    def __init__(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        self.log_dir = log_dir
        self.class_list = class_list
        self.class_map = {i: label_name for i, label_name in enumerate(self.class_list)}
        self.calculate_metrics = calculate_metrics
        self.confidence_threshold = confidence_threshold

        self.epoch_info = defaultdict(list)

        self.logging_keys = ["num_examples", "file_paths", "file_name"]

    def update_epoch_info(self, batch_dict):
        """
        updates epoch info with information from 'batch_dict'
        each batch_dict contains batch information of each forward pass during training

        batch_dict: (dict) the batch information to be added to epoch info
            - total_loss: (list[torch.Tensor]) loss,
            - predictions: (list[torch.Tensor] or list[dict]) model prediction,
            - targets: (torch.Tensor) ground truth,
            - file_name: (list[str]) file name of model input,
            - file_paths: (list[str]) file path of model input,
            - num_examples: (int) number of examples in the batch,
            - scores: confidence score of the prediction,
            - labels: label of the prediction,
        """

        for key in self.logging_keys:
            value = batch_dict[key]
            if not isinstance(value, list):
                value = [value]
            self.epoch_info[key] += value

    def clean_epoch_info(self):
        self.epoch_info = defaultdict(list)

    def get_metrics(self):
        """computes the step statistics and returns them in a dictionary"""
        return self.metric.compute_stats(self.epoch_info)

    def get_loss(self):
        """
        return the average of per sample losses in the epoch.
        """

        total_loss = torch.as_tensor(self.epoch_info["total_loss"])
        num_examples = torch.as_tensor(self.epoch_info["num_examples"])
        average_loss = (total_loss * num_examples).sum() / num_examples.sum()

        return {"average_loss": average_loss.item()}

    def track_progress_stats(self, model_output, **additional_data):
        """tracks stats in the progress using Logger

        Args:
            model_output (dict): model output with two key-value paris
                - loss (torch.Tensor): loss value
                - predictions (torch.Tensor): model prediction
            additional_data (dict): additional data to be tracked in the progress
                - file_paths (list[str]): file path of input images
                - file_name (list[str]): file name of input images
                - num_examples (int): number of examples in the batch
                - targets (torch.Tensor): ground truth
                - labels (torch.Tensor): label of the input image
        """

        with torch.no_grad():

            iteration_stats = {}

            if "loss" in model_output:
                iteration_stats["total_loss"] = [model_output["loss"]]

            iteration_stats["predictions"] = model_output["predictions"]

            iteration_stats.update(additional_data)

            self.update_epoch_info(iteration_stats)

    def prediction_to_dataframe(self):
        pass

    def get_case_study_examples(self):
        pass


class DetectionLogger(Logger):
    def __init__(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.7,
    ):
        super().__init__(
            log_dir,
            class_list,
            calculate_metrics,
            confidence_threshold,
        )
        self.metric = Metric(task="detection", class_map=self.class_map)
        self.matcher = Matcher(
            high_threshold=0.5, low_threshold=0.5, allow_low_quality_matches=False
        )

        self.logging_keys += ["predictions"]
        if self.calculate_metrics:
            self.logging_keys += ["targets"]

    def prediction_to_dataframe(self):
        """
        converts the prediction output to a dataframe
        returns a dataframe with the following columns:
            - file_name: file name of the input image
            - label_name: label name of the prediction
            - x: x coordinate of the bounding box
            - y: y coordinate of the bounding box
            - width: width of the bounding box
            - height: height of the bounding box
            - confidence: confidence score of the prediction
        """
        output = defaultdict(list)

        for batch_pred, batch_file_name in zip(
            self.epoch_info["predictions"], self.epoch_info["file_name"]
        ):
            confident_pred_idx = batch_pred["scores"] >= self.confidence_threshold
            confident_pred_scores = (
                batch_pred["scores"][confident_pred_idx].cpu().numpy()
            )
            confident_pred_boxes = batch_pred["boxes"][confident_pred_idx].cpu().numpy()
            confident_pred_labels = (
                batch_pred["labels"][confident_pred_idx].cpu().numpy()
            )

            output["file_name"] += [batch_file_name]
            output["label_name"] += [
                [self.class_map[pred_cls] for pred_cls in confident_pred_labels]
            ]

            boxes_info = defaultdict(list)
            for pred_box in confident_pred_boxes:
                x, y, w, h = pred_box.tolist()
                boxes_info["x"].append(x)
                boxes_info["y"].append(y)
                boxes_info["width"].append(w)
                boxes_info["height"].append(h)
            output["x"] += [boxes_info["x"]]
            output["y"] += [boxes_info["y"]]
            output["width"] += [boxes_info["width"]]
            output["height"] += [boxes_info["height"]]

            output["confidence"] += [
                [pred_score for pred_score in confident_pred_scores]
            ]

        output = pd.DataFrame(output)
        output = output.apply(pd.Series.explode)
        output["file_name"] = output["file_name"].apply(urllib.parse.unquote_plus)
        output["label_name"] = output["label_name"].fillna("nothing to label")
        output = output.sort_values(["file_name", "label_name"])

        return output

    def get_case_study_examples(self):
        """
        returns the case study examples for each class presented in insight report
        """
        case_study_examples = {}
        for class_idx, class_name in self.class_map.items():
            case_study_examples[class_name] = defaultdict(list)
            for targets, predictions, file_path in zip(
                self.epoch_info["targets"],
                self.epoch_info["predictions"],
                self.epoch_info["file_paths"],
            ):
                targets = targets["boxes"][targets["labels"] == class_idx]

                confidence_indicator = (
                    predictions["scores"] >= self.confidence_threshold
                )
                predictions = predictions["boxes"][
                    confidence_indicator & predictions["labels"] == class_idx
                ]

                # (x1, y1, w, h) -> (x1, y1, x2, y2)
                targets[:, 2] += targets[:, 0]
                targets[:, 3] += targets[:, 1]
                predictions[:, 2] += predictions[:, 0]
                predictions[:, 3] += predictions[:, 1]

                if not targets.numel() and predictions.numel():
                    FP_case = [
                        (file_path, None, each_pred.tolist())
                        for each_pred in predictions
                    ]
                    case_study_examples[class_name]["FP"] += FP_case

                if not predictions.numel() and targets.numel():
                    FN_case = [
                        (file_path, each_target.tolist(), None)
                        for each_target in targets
                    ]
                    case_study_examples[class_name]["FN"] += FN_case

                if targets.numel() and predictions.numel():
                    match_quality_matrix = box_iou(targets, predictions)
                    match_idx_results = self.matcher(match_quality_matrix)

                    for each_pred, match_idx in zip(
                        predictions, match_idx_results.tolist()
                    ):
                        if match_idx > -1:
                            case_study_examples[class_name]["TP"].append(
                                (file_path, targets[match_idx].tolist(), each_pred)
                            )
                        elif match_idx == -1:
                            case_study_examples[class_name]["FP"].append(
                                (file_path, None, each_pred)
                            )

                    unique_match_idxs = match_idx_results.unique()
                    FN_case = []
                    for gt_idx in range(len(targets)):
                        if gt_idx not in unique_match_idxs:
                            FN_case.append((file_path, targets[gt_idx].tolist(), None))
                    case_study_examples[class_name]["FN"] += FN_case

        return case_study_examples


class SingleLabelLogger(Logger):
    def __init__(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        super().__init__(
            log_dir,
            class_list,
            calculate_metrics,
            confidence_threshold,
        )
        self.metric = Metric(task="single_label", class_map=self.class_map)

        self.logging_keys += ["predictions"]
        if self.calculate_metrics:
            self.logging_keys += ["targets"]

    def prediction_to_dataframe(self):
        """
        converts the prediction output to a dataframe
        returns a dataframe with the following columns:
            - file_name: file name of the input image
            - label_name: label name of the prediction
            - confidence: confidence score of the prediction
        """
        output = defaultdict(list)

        for batch_pred, batch_file_name in zip(
            self.epoch_info["predictions"], self.epoch_info["file_name"]
        ):
            prob, idx = torch.max(batch_pred.cpu(), axis=1)
            output["file_name"] += batch_file_name
            output["label_name"] += [
                self.class_map[pred_cls] for pred_cls in idx.numpy().tolist()
            ]
            output["confidence"] += prob.numpy().tolist()

        output = pd.DataFrame(output)
        output["file_name"] = output["file_name"].apply(urllib.parse.unquote_plus)
        output = output.sort_values(["file_name", "label_name"])

        return output

    def get_case_study_examples(self):
        """
        returns the case study examples for each class presented in insight report
        """
        ground_truth = torch.concat(self.epoch_info["targets"], 0)
        score = torch.concat(self.epoch_info["predictions"], 0)
        preds = torch.argmax(score, 1).cpu()
        file_paths = np.concatenate(self.epoch_info["file_paths"])

        case_study_examples = {}
        for idx, class_name in self.class_map.items():
            case_study_examples[class_name] = {}
            case_study_examples[class_name]["TP"] = list(
                file_paths[(ground_truth == idx) & (preds == idx)]
            )
            case_study_examples[class_name]["FP"] = list(
                file_paths[(ground_truth != idx) & (preds == idx)]
            )
            case_study_examples[class_name]["FN"] = list(
                file_paths[(ground_truth == idx) & (preds != idx)]
            )

        return case_study_examples


class MultiLabelLogger(Logger):
    def __init__(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        super().__init__(
            log_dir,
            class_list,
            calculate_metrics,
            confidence_threshold,
        )
        self.metric = Metric(task="multi_label", class_map=self.class_map)

        self.logging_keys += ["predictions"]
        if self.calculate_metrics:
            self.logging_keys += ["targets"]

    def prediction_to_dataframe(self):
        """
        converts the prediction output to a dataframe
        returns a dataframe with the following columns:
            - file_name: file name of the input image
            - label_name: label name of the prediction
            - confidence: confidence score of the prediction
        """
        output = defaultdict(list)

        for batch_pred, batch_file_name in zip(
            self.epoch_info["predictions"], self.epoch_info["file_name"]
        ):
            batch_pred = batch_pred.cpu().numpy()
            output["file_name"] += batch_file_name
            output["label_name"] += [self.class_list for _ in range(len(batch_pred))]
            output["confidence"] += [each_pred.tolist() for each_pred in batch_pred]

        output = pd.DataFrame(output)
        output = output.apply(pd.Series.explode)
        output["file_name"] = output["file_name"].apply(urllib.parse.unquote_plus)
        output = output.sort_values(["file_name", "label_name"])

        return output

    def get_case_study_examples(self):
        """
        returns the case study examples for each class presented in insight report
        """
        ground_truth = torch.concat(self.epoch_info["targets"], 0)
        score = torch.concat(self.epoch_info["predictions"], 0).cpu()
        file_paths = np.concatenate(self.epoch_info["file_paths"])

        case_study_examples = {}
        for idx, class_name in self.class_map.items():
            case_study_examples[class_name] = {}
            y_true = ground_truth[:, idx]
            y_pred = (score[:, idx] >= 0.5).byte()

            case_study_examples[class_name]["TP"] = list(
                file_paths[(y_true == idx) & (y_pred == idx)]
            )
            case_study_examples[class_name]["FP"] = list(
                file_paths[(y_true != idx) & (y_pred == idx)]
            )
            case_study_examples[class_name]["FN"] = list(
                file_paths[(y_true == idx) & (y_pred != idx)]
            )

        return case_study_examples


class SegmentationLogger(Logger):
    def __init__(
        self,
        log_dir,
        class_list,
        calculate_metrics=True,
        confidence_threshold=0.5,
    ):
        super().__init__(
            log_dir,
            class_list,
            calculate_metrics,
            confidence_threshold,
        )
        self.metric = Metric("segmentation", class_map=self.class_map)

        self.logging_keys += ["predictions", "width", "height", "confidences"]
        if self.calculate_metrics:
            self.logging_keys += ["FP", "FN", "TP", "TN", "dice_per_image"]

    def update_epoch_info(self, batch_dict):
        """
        updates epoch info with information from 'batch_dict'
        each batch_dict contains batch information of each forward pass during training

        batch_dict: (dict) the batch information to be added to epoch info
            - total_loss: (torch.Tensor) loss,
            - predictions: (torch.Tensor) model prediction,
            - targets: (torch.Tensor) ground truth,
            - file_name: (list[str]) file name of model input,
            - file_paths: (list[str]) file path of model input,

        epoch_info: (dict)
            - total_loss: list[torch.Tensor],
            - file_name: list[str],
            - TP: list[torch.Tensor] dimension is Batch X Class,
            - FP: list[torch.Tensor] dimension is Batch X Class,
            - FN: list[torch.Tensor] dimension is Batch X Class,
            - TN: list[torch.Tensor] dimension is Batch X Class,
        """

        ori_predictions = batch_dict.pop("predictions").cpu()
        predictions = (ori_predictions > self.confidence_threshold).float()

        if self.calculate_metrics:
            targets = batch_dict.pop("targets")

            FP, FN, TP, TN = numeric_score(predictions, targets, dim=(2, 3))
            batch_dict["dice_per_image"] = [
                np.divide(2 * TP, 2 * TP + FP + FN).tolist()
            ]
            batch_dict.update({"FP": FP, "FN": FN, "TP": TP, "TN": TN})

        batch_pred_mask = []
        for each_pred, each_width, each_height in zip(
            predictions, batch_dict["width"][0], batch_dict["height"][0]
        ):
            each_pred = each_pred.byte().numpy()
            encoded_mask = []
            for idx in range(len(self.class_list)):
                mask = cv2.resize(each_pred[idx], (each_width, each_height))
                encoded_mask.append(encode(mask))
            batch_pred_mask.append(encoded_mask)
        batch_dict["predictions"] = [batch_pred_mask]
        batch_dict["confidences"] = [
            ori_predictions.amax(dim=(2, 3)).numpy().tolist()
        ]

        return super().update_epoch_info(batch_dict)

    def prediction_to_dataframe(self):
        """
        converts the prediction output to a dataframe
        returns a dataframe with the following columns:
            - file_name: file name of the input image
            - label_name: label name of the prediction
            - width: width of the input image
            - height: height of the input image
            - mask: encoded mask of the prediction
            - predictions: model prediction
            - confidence: confidence score of the prediction
            - dice (if calculate metrics): dice score of the prediction
        """
        output = defaultdict(list)

        for (
            batch_file_name,
            batch_width,
            batch_height,
            batch_pred,
            batch_confidence,
        ) in zip(
            self.epoch_info["file_name"],
            self.epoch_info["width"],
            self.epoch_info["height"],
            self.epoch_info["predictions"],
            self.epoch_info["confidences"],
        ):

            output["file_name"] += batch_file_name
            output["label_name"] += [
                self.class_list for _ in range(len(batch_file_name))
            ]
            output["width"] += batch_width
            output["height"] += batch_height

            output["mask"] += batch_pred
            output["confidence"] += batch_confidence

        if self.calculate_metrics:
            for each_dice in self.epoch_info["dice_per_image"]:
                output["dice"] += each_dice

        output = pd.DataFrame(output)
        output = output.apply(pd.Series.explode)
        output["file_name"] = output["file_name"].apply(urllib.parse.unquote_plus)
        if "dice" not in output.columns:
            output["dice"] = None

        output = output.sort_values(["file_name", "label_name"])

        return output

    def get_case_study_examples(self):
        """
        returns the case study examples for each class presented in insight report
        """
        TPs = torch.cat(self.epoch_info["TP"], 0)
        TNs = torch.cat(self.epoch_info["TN"], 0)
        FPs = torch.cat(self.epoch_info["FP"], 0)
        FNs = torch.cat(self.epoch_info["FN"], 0)
        file_paths = np.concatenate(self.epoch_info["file_paths"])

        case_study_examples = {}
        for idx, class_name in self.class_map.items():
            case_study_examples[class_name] = {}
            TP, TN, FP, FN = TPs[:, idx], TNs[:, idx], FPs[:, idx], FNs[:, idx]
            iou = torch.divide(TP, TP + FP + FN)
            case_study_examples[class_name]["TP"] = list(file_paths[iou >= 0.5])
            case_study_examples[class_name]["FP"] = list(file_paths[iou < 0.25])
            case_study_examples[class_name]["FN"] = list(
                file_paths[torch.argwhere(torch.isnan(iou)).squeeze()]
            )

        return case_study_examples
