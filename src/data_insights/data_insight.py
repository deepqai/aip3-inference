import os

import numpy as np
import pandas as pd

from constants.preprocess import DATA_INSIGHT_FOLDER


class DataInsight:
    def __init__(self, datadir, class_map, label_insight_extractor):
        self.datadir = datadir
        self.class_map = class_map
        self.class_list = list(self.class_map.keys())
        self.label_insight_extractor = label_insight_extractor(self.class_list)

    def count_stat_to_dataframe_dict(self, label_stats, **kwargs):
        stats = label_stats["stats"]
        count = [stats.get(class_name, 0) for class_name in self.class_list]

        df = pd.DataFrame(
            {"name": self.class_list, "name_for_pdf": self.class_list, "count": count}
        )

        df["percentage"] = (
            ((df["count"] / label_stats["total_num"]) * 100).fillna(0).round(2)
        )
        df.loc[len(df)] = {
            "name": "total",
            "name_for_pdf": "TOTAL",
            "count": label_stats["total_num"],
            "percentage": 100.0,
        }

        return {"all_labels": df}

    def interval_stat_to_dataframe_dict(self, label_stats, **kwargs):
        stats = label_stats["stats"]

        bins = kwargs["bins"]
        interval_names = kwargs["interval_names"].keys()
        interval_names_for_pdf = kwargs["interval_names"].values()

        dataframe_dict = {}
        for class_name in self.class_list + ["all_labels"]:
            # add negative sign to each bin so that we can use np.histrogram in our case (exclude lower_bound, include upper_bound)
            negative_bins = sorted([-each_bin for each_bin in bins])
            negative_stats = [-each_stat for each_stat in stats[class_name]]
            num_in_negative_interval = np.histogram(negative_stats, bins=negative_bins)[
                0
            ].tolist()

            num_in_interval = num_in_negative_interval[::-1]

            df = pd.DataFrame(
                {
                    "name": interval_names,
                    "name_for_pdf": interval_names_for_pdf,
                    "count": num_in_interval,
                }
            )
            df["percentage"] = (
                (df["count"] / df["count"].sum() * 100).fillna(0).round(2)
            )
            df.loc[len(df)] = {
                "name": "total",
                "name_for_pdf": "TOTAL",
                "count": df["count"].sum(),
                "percentage": df["percentage"].sum(),
            }

            dataframe_dict[class_name] = df

        return dataframe_dict

    def run(self, dataset_name, df):
        assert os.path.isdir(
            self.datadir
        ), f"Data directory {self.datadir} is not a valid directory."

        dataset_insight_folder = os.path.join(
            self.datadir, DATA_INSIGHT_FOLDER, dataset_name
        )

        save_img_folder = os.path.join(dataset_insight_folder, "img")
        os.makedirs(save_img_folder, exist_ok=True)

        save_csv_folder = os.path.join(dataset_insight_folder, "csv")
        os.makedirs(save_csv_folder, exist_ok=True)

        original_stats = self.label_insight_extractor.run(df)
        data_statistics = {}
        for stat_name, stat_data in original_stats.items():
            data_statistics[stat_name] = {}

            stat_info = self.statistics[stat_name]
            stat_type = stat_info["type"]
            handler, handler_params = stat_info["handler"], stat_info["handler_params"]

            stat_df_dict = handler(stat_data, **handler_params)

            data_statistics[stat_name]["labels"] = []
            for class_name, stat_df in stat_df_dict.items():
                stat_df_for_pdf = stat_df.loc[
                    :, ["name_for_pdf", "count", "percentage"]
                ]
                stat_df_for_pdf = stat_df_for_pdf.rename(
                    columns={"name_for_pdf": "name"}
                )

                save_csv_path = f"{save_csv_folder}/{stat_name}_{class_name}.csv"
                stat_df_for_pdf.to_csv(save_csv_path, index=False)

                stat_df_for_api = stat_df.loc[:, ["name", "count", "percentage"]]
                if stat_type == "count":
                    data_statistics[stat_name]["labels"] = stat_df_for_api.to_dict(
                        orient="records"
                    )
                elif stat_type == "interval":
                    tmp_dict = stat_df_for_api.set_index("name").to_dict("index")
                    tmp_dict["name"] = class_name
                    data_statistics[stat_name]["labels"].append(tmp_dict)

        # for detection and segmentation
        if len(data_statistics) > 1:
            return data_statistics

        return list(data_statistics.values())[0]


class SingleLabelDataInsight(DataInsight):
    def __init__(self, args, class_map, label_insight_extractor):
        super().__init__(args, class_map, label_insight_extractor)

        self.statistics = {
            "label_distribution": {
                "type": "count",
                "handler": self.count_stat_to_dataframe_dict,
                "handler_params": {},
                "show_full_len": False,
            }
        }


class MultiLabelDataInsight(DataInsight):
    def __init__(self, args, class_map, label_insight_extractor):
        super().__init__(args, class_map, label_insight_extractor)

        self.statistics = {
            "label_distribution": {
                "type": "count",
                "handler": self.count_stat_to_dataframe_dict,
                "handler_params": {},
                "show_full_len": True,
            }
        }


class DetectionDataInsight(DataInsight):
    def __init__(self, args, class_map, label_insight_extractor):
        super().__init__(args, class_map, label_insight_extractor)
        self.class_list = self.class_list[1:]

        self.box_per_image_interval_names = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five_ten": "5-10",
            "larger_than_ten": ">10",
        }
        self.box_ratio_interval_names = {
            "narrow": "Narrow",
            "semi_narrow": "Semi-Narrow",
            "square": "Square",
            "semi_wide": "Semi-Wide",
            "wide": "Wide",
        }
        self.box_area_interval_names = {
            "small": "Small",
            "medium": "Medium",
            "large": "Large",
        }

        self.statistics = {
            "image_wise": {
                "type": "count",
                "handler": self.count_stat_to_dataframe_dict,
                "handler_params": {},
                "show_full_len": True,
            },
            "box_wise": {
                "type": "count",
                "handler": self.count_stat_to_dataframe_dict,
                "handler_params": {},
                "show_full_len": False,
            },
            "box_per_image": {
                "type": "interval",
                "handler": self.interval_stat_to_dataframe_dict,
                "handler_params": {
                    "bins": [-1, 0, 1, 2, 3, 4, 10, float("inf")],
                    "interval_names": self.box_per_image_interval_names,
                },
                "show_full_len": False,
            },
            "box_ratio": {
                "type": "interval",
                "handler": self.interval_stat_to_dataframe_dict,
                "handler_params": {
                    "bins": [0, 1 / 2, 4 / 5, 5 / 4, 2, float("inf")],
                    "interval_names": self.box_ratio_interval_names,
                },
                "show_full_len": False,
            },
            "box_area": {
                "type": "interval",
                "handler": self.interval_stat_to_dataframe_dict,
                "handler_params": {
                    "bins": [0, 32 * 32, 96 * 96, float("inf")],
                    "interval_names": self.box_area_interval_names,
                },
                "show_full_len": False,
            },
        }


class SegmentationDataInsight(DataInsight):
    def __init__(self, args, class_map, label_insight_extractor):
        super().__init__(args, class_map, label_insight_extractor)

        self.mask_coverage_interval_names = {
            "zero": "0%",
            "zero_ten": "0% - 10%",
            "ten_twenty": "10% - 20%",
            "twenty_thirty": "20% - 30%",
            "thirty_forty": "30% - 40%",
            "forty_fifty": "40% - 50%",
            "fifty_sixty": "50% - 60%",
            "sixty_seventy": "60% - 70%",
            "seventy_eighty": "70% - 80%",
            "eighty_ninty": "80% - 90%",
            "ninty_hundred": "90% - 100%",
        }

        self.statistics = {
            "label_distribution": {
                "type": "count",
                "handler": self.count_stat_to_dataframe_dict,
                "handler_params": {},
                "show_full_len": True,
            },
            "mask_coverage_distribution": {
                "type": "interval",
                "handler": self.interval_stat_to_dataframe_dict,
                "handler_params": {
                    "bins": [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    "interval_names": self.mask_coverage_interval_names,
                },
                "show_full_len": False,
            },
        }
