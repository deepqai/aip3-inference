from abc import ABC, abstractmethod


class LabelInsightExtractor(ABC):
    def __init__(self, class_list):
        self.class_list = class_list

    @abstractmethod
    def run(self, df):
        pass


class MultiLabelInsightExtractor(LabelInsightExtractor):
    def __init__(self, class_list):
        super().__init__(class_list)

    def run(self, df):
        label_distribution = {}

        grouped_df = df.groupby("label")

        label_distribution["stats"] = {class_name: 0 for class_name in self.class_list}
        label_distribution["stats"].update(
            grouped_df["binary_label"].apply(lambda x: sum(x > 0.5)).to_dict()
        )
        label_distribution["total_num"] = df["file_name"].nunique()

        return {"label_distribution": label_distribution}


class SingleLabelInsightExtractor(LabelInsightExtractor):
    def __init__(self, class_list):
        super().__init__(class_list)

    def run(self, df):
        label_distribution = {}
        label_distribution["stats"] = {class_name: 0 for class_name in self.class_list}
        label_distribution["stats"].update(df["label"].value_counts().to_dict())
        label_distribution["total_num"] = df["file_name"].nunique()

        return {"label_distribution": label_distribution}


class DetectionInsightExtractor(LabelInsightExtractor):
    def __init__(self, class_list):
        self.no_annotation_label = class_list[0]
        self.class_list = class_list[1:]

    def _calculate_box_per_image(self, series):
        series.value_counts().tolist()

    def run(self, df):
        process_df = df[df["label"] != self.no_annotation_label]
        total_image = df["file_name"].nunique()
        total_box = len(process_df["label"])

        grouped_process_df = process_df.groupby("label")

        # handle image_wise label distribution
        image_wise_label_distribution = {}
        image_wise_label_distribution["stats"] = {
            class_name: 0 for class_name in self.class_list
        }
        image_wise_label_distribution["stats"].update(
            grouped_process_df["file_name"].nunique().to_dict()
        )
        image_wise_label_distribution["total_num"] = total_image

        # handle box_wise label distribution
        box_wise_label_distribution = {}
        box_wise_label_distribution["stats"] = {
            class_name: 0 for class_name in self.class_list
        }
        box_wise_label_distribution["stats"].update(process_df["label"].value_counts())
        box_wise_label_distribution["total_num"] = total_box

        # handle box_per_image distribution
        box_per_image_distribution = {}
        box_per_image_distribution["stats"] = {
            class_name: [0] * total_image for class_name in self.class_list
        }  # make sure all dataset splits have same label set
        box_per_image_distribution["stats"].update(
            grouped_process_df["file_name"]
            .agg(lambda series: series.value_counts().tolist())
            .to_dict()
        )
        box_per_image_distribution["stats"]["all_labels"] = (
            process_df["file_name"].value_counts().tolist()
        )
        box_per_image_distribution["stats"] = {
            k: v + [0] * (total_image - len(v))
            for k, v in box_per_image_distribution["stats"].items()
        }

        # handle box_ratio distribution
        box_ratio_distribution = {}
        box_ratio_distribution["stats"] = {
            class_name: [-1] for class_name in self.class_list
        }  # make sure all dataset splits have same label set
        box_ratio_distribution["stats"].update(
            grouped_process_df["bbox_w_div_h"].agg(list)
        )
        box_ratio_distribution["stats"]["all_labels"] = process_df[
            "bbox_w_div_h"
        ].tolist()

        # handle box_area distribution
        box_area_distribution = {}
        box_area_distribution["stats"] = {
            class_name: [-1] for class_name in self.class_list
        }  # make sure all dataset splits have same label set
        box_area_distribution["stats"].update(grouped_process_df["bbox_area"].agg(list))
        box_area_distribution["stats"]["all_labels"] = process_df["bbox_area"].tolist()

        stats = {
            "image_wise": image_wise_label_distribution,
            "box_wise": box_wise_label_distribution,
            "box_per_image": box_per_image_distribution,
            "box_ratio": box_ratio_distribution,
            "box_area": box_area_distribution,
        }

        return stats


class SegmentationInsightExtractor(LabelInsightExtractor):
    def __init__(self, class_list):
        super().__init__(class_list)

    def run(self, df):
        label_distribution = {}
        process_df = df[df["mask_area_percentage"] > 0]
        label_distribution["stats"] = process_df["label"].value_counts().to_dict()
        label_distribution["total_num"] = df["file_name"].nunique()

        mask_coverage_distribution = {}
        mask_coverage_distribution["stats"] = (
            df.groupby("label")["mask_area_percentage"].agg(list).to_dict()
        )
        mask_coverage_distribution["stats"]["all_labels"] = df[
            "mask_area_percentage"
        ].tolist()

        return {
            "label_distribution": label_distribution,
            "mask_coverage_distribution": mask_coverage_distribution,
        }
