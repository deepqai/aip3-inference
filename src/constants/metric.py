REALTIME_STATISTICS_NAME_MAPPING = {
    "single_label": {
        "score": "accuracy",
    },
    "multi_label": {
        "score": "accuracy",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
    },
    "detection": {
        "score": "ap",
        "ap50": "ap50",
        "ap75": "ap75",
    },
    "segmentation": {
        "score": "dice",
        "iou": "iou",
        "accuracy": "accuracy",
    },
}
