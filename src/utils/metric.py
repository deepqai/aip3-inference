from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from .metric_utils import (
    CustomMeanAveragePrecision,
    curve_approximation,
    drop_intermediate,
)


def numeric_score(prediction, groundtruth, *args, **kwargs):
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    FP = torch.sum((prediction == 1) & (groundtruth == 0), *args, **kwargs)
    FN = torch.sum((prediction == 0) & (groundtruth == 1), *args, **kwargs)
    TP = torch.sum((prediction == 1) & (groundtruth == 1), *args, **kwargs)
    TN = torch.sum((prediction == 0) & (groundtruth == 0), *args, **kwargs)
    return FP, FN, TP, TN


class Metric:
    def __init__(self, task, class_map):
        self.task = task
        self.class_map = class_map

    def compute_stats(self, epoch_info):
        # self._check_formate(batch_info)
        if self.task == "multi_label":
            return self._compute_multi_label_metric(epoch_info)
        elif self.task == "single_label":
            return self._compute_single_label_metric(epoch_info)
        elif self.task == "detection":
            return self._compute_detection_metric(epoch_info)
        elif self.task == "segmentation":
            return self._compute_segmentation_metric(epoch_info)
        else:
            raise ValueError()

    def _check_formate(self, epoch_info):
        pass

    def _truncate(self, x, decimals=0):
        scale = 10**decimals
        return np.trunc(x * scale) / scale

    def _compute_multi_label_metric(self, epoch_info):
        ground_truth = torch.concat(epoch_info["targets"], 0).cpu().numpy()
        score = torch.concat(epoch_info["predictions"], 0).cpu().numpy()

        thr = 0.5

        metric = {}
        for idx, class_name in self.class_map.items():
            y_true = ground_truth[:, idx]
            y_score = score[:, idx]

            fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)
            rocauc = auc(fpr, tpr)
            fpr, tpr = curve_approximation(fpr, tpr)

            pre, rec, _ = precision_recall_curve(y_true, y_score)
            prauc = auc(rec, pre)
            pre, rec = drop_intermediate(pre, rec)
            pre, rec = curve_approximation(pre, rec)
            pre, rec = np.flip(pre), np.flip(rec)

            y_predict = (y_score >= thr).astype(int)
            TN, FP, FN, TP = confusion_matrix(y_true, y_predict, labels=[0, 1]).ravel()

            metric[class_name] = OrderedDict(
                {
                    "roc_auc": (rocauc * 100),
                    "fpr": (fpr * 100),  # for roc-curve
                    "tpr": (tpr * 100),  # for roc-curve
                    "pr_auc": (prauc * 100),
                    "pr_precision": (pre * 100),  # for pr-curve
                    "pr_recall": (rec * 100),  # for pr-curve
                    "tp": TP,
                    "tn": TN,
                    "fp": FP,
                    "fn": FN,
                    "accuracy": (np.divide(TP + TN, TP + FP + TN + FN) * 100),
                    "precision": (np.divide(TP, TP + FP) * 100),
                    "recall": (np.divide(TP, TP + FN) * 100),
                    "specificity": (np.divide(TN, TN + FP) * 100),
                    "f1": (np.divide(2 * TP, 2 * TP + FP + FN) * 100),
                }
            )

        # average
        average_metric_key = [
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
        ]
        metric["average"] = OrderedDict({})
        for key in average_metric_key:
            metric_each_class = [
                metric[class_name][key] for class_name in self.class_map.values()
            ]
            metric["average"][key] = np.nanmean(metric_each_class)

        # truncate to 3 decimal or round to int
        for label, metric_class in metric.items():
            for metric_key, metric_value in metric_class.items():
                if metric_key in ["TP", "FP", "FN", "TN"]:
                    metric_value = np.around(metric_value).astype(np.int)
                else:
                    metric_value = self._truncate(metric_value, 3)
                metric[label][metric_key] = (
                    metric_value.item()
                    if metric_value.size == 1
                    else metric_value.tolist()
                )

        return metric

    def _compute_single_label_metric(self, epoch_info):
        ground_truth = torch.concat(epoch_info["targets"], 0).cpu().numpy()
        score = torch.concat(epoch_info["predictions"], 0).cpu().numpy()
        num_classes = score.shape[1]

        top1_predictions = np.argmax(score, 1)
        confmat = confusion_matrix(
            ground_truth, top1_predictions, labels=np.arange(num_classes)
        )
        confmat = confmat.transpose()

        metric = {}
        for idx, class_name in self.class_map.items():
            y_true = ground_truth
            y_score = score[:, idx]

            fpr, tpr, _ = roc_curve(
                y_true, y_score, pos_label=idx, drop_intermediate=True
            )
            rocauc = auc(fpr, tpr)
            fpr, tpr = curve_approximation(fpr, tpr)

            pre, rec, _ = precision_recall_curve(y_true, y_score, pos_label=idx)
            prauc = auc(rec, pre)
            pre, rec = drop_intermediate(pre, rec)
            pre, rec = curve_approximation(pre, rec)
            pre, rec = np.flip(pre), np.flip(rec)

            TP = confmat[idx, idx]
            FP, FN = np.sum(confmat[idx, :]) - TP, np.sum(confmat[:, idx]) - TP
            TN = np.sum(confmat) - TP - FP - FN

            metric[class_name] = OrderedDict(
                {
                    "roc_auc": (rocauc * 100),
                    "fpr": (fpr * 100),  # for roc-curve
                    "tpr": (tpr * 100),  # for roc-curve
                    "pr_auc": (prauc * 100),
                    "pr_precision": (pre * 100),  # for pr-curve
                    "pr_recall": (rec * 100),  # for pr-curve
                    "accuracy": (np.divide(TP + TN, TP + FP + TN + FN) * 100),
                    "precision": (np.divide(TP, TP + FP) * 100),
                    "recall": (np.divide(TP, TP + FN) * 100),
                    "specificity": (np.divide(TN, TN + FP) * 100),
                    "f1": (np.divide(2 * TP, 2 * TP + FP + FN) * 100),
                }
            )

        # average
        average_metric_key = [
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "specificity",
            "f1",
        ]
        metric["average"] = OrderedDict({})
        for key in average_metric_key:
            metric_each_class = [
                metric[class_name][key] for class_name in self.class_map.values()
            ]
            metric["average"][key] = np.nanmean(metric_each_class)
        metric["average"]["accuracy"] = (np.trace(confmat) / confmat.sum()) * 100

        # truncate to 3 decimal
        for label, metric_class in metric.items():
            for metric_key, metric_value in metric_class.items():
                metric_value = self._truncate(metric_value, 3)
                metric[label][metric_key] = (
                    metric_value.item()
                    if metric_value.size == 1
                    else metric_value.tolist()
                )

        metric["confusion_matrix"] = {"matrix": confmat.astype(np.int).tolist()}

        return metric

    def _compute_detection_metric(self, epoch_info):
        """
        computes the mean average precision (mAP) for detection

        See https://torchmetrics.readthedocs.io/en/latest/references/modules.html#detection-metrics for details.
        Current requirments ask for mAP, mAP50, and mAP75, so these are returned
        """

        predictions = epoch_info.get("predictions", [])
        targets = epoch_info.get("targets", [])

        metric_MAP = CustomMeanAveragePrecision(box_format="xywh")
        # TODO add automatic-sync to other metric
        metric_MAP._to_sync = False  # already sync by logger.
        metric_MAP.update(predictions, targets)
        map_scores = metric_MAP.compute()

        # get mAP, mAP50 and mAP75
        metrics = OrderedDict({})
        metrics["summary"] = OrderedDict(
            {
                "ap": {
                    "iou_fifty": map_scores["map_50"] * 100,
                    "iou_seventy_five": map_scores["map_75"] * 100,
                    "average": map_scores["map"] * 100,
                },
                "ap_small": {
                    "iou_fifty": map_scores["map_small_50"] * 100,
                    "iou_seventy_five": map_scores["map_small_75"] * 100,
                    "average": map_scores["map_small"] * 100,
                },
                "ap_medium": {
                    "iou_fifty": map_scores["map_medium_50"] * 100,
                    "iou_seventy_five": map_scores["map_medium_75"] * 100,
                    "average": map_scores["map_medium"] * 100,
                },
                "ap_large": {
                    "iou_fifty": map_scores["map_large_50"] * 100,
                    "iou_seventy_five": map_scores["map_large_75"] * 100,
                    "average": map_scores["map_large"] * 100,
                },
                "ar_small": {
                    "average": map_scores["mar_small"] * 100,
                },
                "ar_medium": {
                    "average": map_scores["mar_medium"] * 100,
                },
                "ar_large": {
                    "average": map_scores["mar_large"] * 100,
                },
                "ar_max_one": {
                    "average": map_scores["mar_1"] * 100,
                },
                "ar_max_10": {
                    "average": map_scores["mar_10"] * 100,
                },
                "ar_max_100": {
                    "average": map_scores["mar_100"] * 100,
                },
            }
        )
        metrics["average"] = OrderedDict(
            {
                "ap": map_scores["map"] * 100,
                "ap50": map_scores["map_50"] * 100,
                "ap75": map_scores["map_75"] * 100,
            }
        )

        for idx in map_scores["pre"].keys():
            pre = map_scores["pre"][idx]
            rec = map_scores["rec"][idx]
            pre, rec = drop_intermediate(pre, rec)
            pre, rec = curve_approximation(pre, rec)

            tpr = map_scores["rec"][idx]
            fppi = map_scores["fp_per_image"][idx]
            tpr, fppi = drop_intermediate(tpr, fppi)
            tpr, fppi = curve_approximation(tpr, fppi)

            metrics[self.class_map[idx]] = OrderedDict(
                {
                    "precision_fifty": pre * 100,
                    "tpr_fifty": tpr * 100,
                    "recall_fifty": rec * 100,
                    "fppi_fifty": fppi,
                }
            )

        # truncate to 3 decimal
        for label, metric_class in metrics.items():
            for metric_key, metric_value in metric_class.items():
                if isinstance(metric_value, dict):
                    for sub_metric_key, sub_metric_value in metric_value.items():
                        # 　convert -100 to NaN
                        sub_metric_value[sub_metric_value == -100] = torch.nan

                        sub_metric_value = self._truncate(
                            sub_metric_value.numpy().astype(np.float), 3
                        )
                        metrics[label][metric_key][sub_metric_key] = sub_metric_value
                else:
                    # 　convert -100 to NaN
                    metric_value[metric_value == -100] = torch.nan

                    metric_value = self._truncate(
                        metric_value.numpy().astype(np.float), 3
                    )
                    metrics[label][metric_key] = (
                        metric_value.item()
                        if metric_value.size == 1
                        else metric_value.tolist()
                    )

        return metrics

    def _compute_segmentation_metric(self, batch_info):
        # get information
        TPs = torch.cat(batch_info["TP"], 0).cpu().numpy()
        TNs = torch.cat(batch_info["TN"], 0).cpu().numpy()
        FPs = torch.cat(batch_info["FP"], 0).cpu().numpy()
        FNs = torch.cat(batch_info["FN"], 0).cpu().numpy()

        # compute metric
        metric = {}
        for idx, class_name in self.class_map.items():
            TP, TN, FP, FN = TPs[:, idx], TNs[:, idx], FPs[:, idx], FNs[:, idx]
            total_count = TP + TN + FP + FN

            metric[class_name] = OrderedDict(
                {
                    "fp": np.nanmean(np.divide(FP, total_count)) * 100,
                    "fn": np.nanmean(np.divide(FN, total_count)) * 100,
                    "tp": np.nanmean(np.divide(TP, total_count)) * 100,
                    "tn": np.nanmean(np.divide(TN, total_count)) * 100,
                    "accuracy": np.nanmean(np.divide(TP + TN, total_count)) * 100,
                    "iou": np.nanmean(np.divide(TP, TP + FP + FN)) * 100,
                    "precision": np.nanmean(np.divide(TP, TP + FP)) * 100,
                    "specificity": np.nanmean(np.divide(TN, TN + FP)) * 100,
                    "recall": np.nanmean(np.divide(TP, TP + FN)) * 100,
                    "dice": np.nanmean(np.divide(2 * TP, 2 * TP + FP + FN)) * 100,
                }
            )

        # average
        metric["average"] = OrderedDict({})
        for key in metric[self.class_map[0]].keys():
            metric_each_class = [
                metric[class_name][key] for class_name in self.class_map.values()
            ]
            metric["average"][key] = np.nanmean(metric_each_class)

        # truncate to 3 decimal
        for label, metric_class in metric.items():
            for metric_key, metric_value in metric_class.items():
                metric_value = self._truncate(metric_value, 3)
                metric[label][metric_key] = metric_value.item()

        return metric
