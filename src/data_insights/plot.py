import os
from pathlib import Path
from random import sample

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from cycler import cycler
from matplotlib import colors
from matplotlib.font_manager import FontProperties as font
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay

matplotlib.use("Agg")

COLOR_MAP = [
    "#03A9F4",
    "#00C853",
    "#FF9800",
    "#E040FB",
    "#EF9A9A",
    "#90CAF9",
    "#B2FF59",
    "#FFEB3B",
    "#9C27B0",
    "#BF360C",
    "#C8C1FF",
    "#EFADCA",
    "#CBBEB9",
    "#0052FF",
    "#FF7979",
    "#45BAAE",
    "#D6A98A",
    "#A1C4A3",
    "#95B9E2",
    "#F8E68C",
    "#614DFF",
]


def horizontal_bar_chart(stat_dict, save_path, y_limit, show_full_len=False):
    """
    Plots a horizontal bar chart as per AIP format
    Args:
        stat_dict: (dict) containing the bar name and value (value given in percent value, not decimal)
        save_path: (str) directory to save the plot figure
        show_full_len: (bool) show the bar out of 100% (will add negative remaining to fill to 100%)
    """

    GenShinGothic_font = font(
        fname=os.path.join(
            os.path.dirname(__file__),
            "supporting_files",
            "font",
            "GenShinGothic-Regular.ttf",
        )
    )

    # preprocess the extracted info
    category_names = []
    category_percentages = []
    for name, percentage in stat_dict.items():
        if name == "total":
            continue
        category_names.append(name if len(name) < 19 else f"{name[:16]}...")
        category_percentages.append(percentage)

    # generate plot
    fig_height = 0.5 * len(category_names) + 0.5
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, fig_height), dpi=120)

    if show_full_len:
        ax.barh(
            category_names,
            [100 for _ in category_names],
            height=0.5,
            color="#FF9800",
            label="Negative",
        )
    ax.barh(
        category_names,
        category_percentages,
        height=0.5,
        color="#03A9F4",
        label="Positive",
    )  # plot per class count percent
    if not show_full_len:
        ax.set_xlim([0, y_limit])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))

    # add percentage info to each bar
    y_offset = 0.02
    print_split = 5 if show_full_len else max(category_percentages, default=0) * 0.05
    for i, pct in enumerate(category_percentages):
        if pct == 0:
            str_pct = "0%"
        elif pct < 1:
            str_pct = "<1%"
        else:
            str_pct = f"{int(round(pct))}%"  # round percentage to closes integer

        if int(round(pct)) > print_split:
            plt.text(
                pct / 2, i + y_offset, str_pct, color="white", ha="center", va="center"
            )
        else:
            plt.text(
                pct, i + y_offset, " " + str_pct, color="black", ha="left", va="center"
            )

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_frame_on(False)
    ax.grid(axis="x", color="#E6E6E6")

    for label in ax.get_yticklabels():
        label.set_fontproperties(GenShinGothic_font)
        label.set_fontweight("heavy")

    ax.tick_params(axis="both", which="both", length=0, labelcolor="#11456C", pad=8)

    ax.set_axisbelow(True)
    if show_full_len:
        ax.legend(
            frameon=False,
            loc="upper right",
            ncol=2,
            labelcolor="#11456C",
            bbox_to_anchor=(1, -0.5 / fig_height),
            handlelength=1,
            handleheight=1,
        )

    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_model_performance(
    plot_data, axis_name, average_performance, dash_line=None, save_path=None
):
    """
    Plots a training curve for training progress visualization purposes
    Args:
        data_dict: (dict) contains the keys to plot for ('train' -> Training, 'batch_inference' -> Validation)
        save_path: (str) path to save the plot figure
    """

    GenShinGothic_font = font(
        fname=os.path.join(
            os.path.dirname(__file__),
            "supporting_files",
            "font",
            "GenShinGothic-Regular.ttf",
        )
    )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.set_prop_cycle(cycler(color=COLOR_MAP))

    # plot curves
    for class_name, data in plot_data.items():
        ax.plot(data[0], data[1], label=f"{class_name}{data[2]}")

    average_performance_name, average_performance_value = average_performance
    plt.text(
        1,
        1.04,
        f"{average_performance_name}={average_performance_value}%",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, -0.04),
        handlelength=1,
        handleheight=1,
        prop=GenShinGothic_font,
    )
    ax.grid(axis="y", color="#E6E6E6")
    ax.set_xlabel(axis_name[0], loc="right")
    ax.set_xlim(left=0)
    ax.set_ylabel(axis_name[1], ha="left", y=1.04, rotation=0)
    ax.set_ylim(0, 100)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


# confusion matrix
def plot_confusion_matrix(confusion_matrix, class_list, save_path=None):
    GenShinGothic_font = font(
        fname=os.path.join(
            os.path.dirname(__file__),
            "supporting_files",
            "font",
            "GenShinGothic-Regular.ttf",
        )
    )

    fig, ax = plt.subplots(tight_layout=True)

    rgb = colors.to_rgb("#11456C")
    color = [(1, 1, 1), (rgb[0], rgb[1], rgb[2])]
    cmap = colors.LinearSegmentedColormap.from_list("cmap", color)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=class_list
    )
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_label_position("right")

    image = disp.plot(cmap=cmap, ax=ax)

    colorbar = image.im_.colorbar
    colorbar.set_label("Number of Examples", rotation=-90, labelpad=15)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        font=GenShinGothic_font,
    )
    plt.setp(ax.get_yticklabels(), font=GenShinGothic_font)
    plt.xlabel("True Label")
    plt.ylabel("Predictions", rotation=-90, va="bottom")

    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_case_studies(task_type, plot_data, n_row, n_col, datadir, save_path=None):
    n_samples = int(n_row * n_col)
    samples = (
        sample(plot_data, n_samples) if len(plot_data) > n_samples else plot_data
    )  # sample bboxes

    n_row = max(
        1, int(np.ceil(len(samples) / n_col))
    )  # reduce number of rows needed to plot if too less bboxes
    scale = 4
    fig = plt.figure(figsize=(n_col * scale, n_row * scale))

    os.makedirs(Path(save_path).parent, exist_ok=True)

    # handle empty case
    if len(samples) == 0:
        fig.text(
            0.5, 0.5, "No case studies to show", ha="center", va="center", fontsize=20
        )
        plt.axis("off")
        plt.savefig(save_path, transparent=True)
        return

    # create subplot
    gs = fig.add_gridspec(n_row, n_col, hspace=0.05, wspace=0.05)
    subplot_list_2d = gs.subplots(squeeze=False)

    img_idx = 0
    for i in range(len(subplot_list_2d)):
        for j in range(len(subplot_list_2d[i])):

            subplot_list_2d[i][j].axis("off")

            if img_idx >= len(samples):
                continue

            # read data from extracted info
            if task_type != "detection":
                img_path = samples[img_idx]
                gt_bbox = pred_bbox = None
            else:
                img_path = samples[img_idx][0]
                gt_bbox = samples[img_idx][1]
                pred_bbox = samples[img_idx][2]

            img_path = os.path.join(datadir, "image", img_path)

            # read image & annotate
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if pred_bbox is not None and gt_bbox is not None:
                cv2.rectangle(
                    img,
                    (int(pred_bbox[0]), int(pred_bbox[1])),
                    (int(pred_bbox[2]), int(pred_bbox[3])),
                    (117, 249, 76),
                    3,
                )
                cv2.rectangle(
                    img,
                    (int(gt_bbox[0]), int(gt_bbox[1])),
                    (int(gt_bbox[2]), int(gt_bbox[3])),
                    (228, 78, 40),
                    3,
                )
            elif gt_bbox is not None:
                cv2.rectangle(
                    img,
                    (int(gt_bbox[0]), int(gt_bbox[1])),
                    (int(gt_bbox[2]), int(gt_bbox[3])),
                    (228, 78, 40),
                    3,
                )
            elif pred_bbox is not None:
                cv2.rectangle(
                    img,
                    (int(pred_bbox[0]), int(pred_bbox[1])),
                    (int(pred_bbox[2]), int(pred_bbox[3])),
                    (117, 249, 76),
                    3,
                )

            subplot_list_2d[i][j].imshow(img)
            img_idx += 1

    if task_type == "detection":
        legend_element = [
            Line2D(
                [0], [0], color=(228 / 255, 78 / 255, 40 / 255), label="Ground Truth"
            ),
            Line2D(
                [0], [0], color=(117 / 255, 249 / 255, 76 / 255), label="Prediction"
            ),
        ]
        fig.legend(handles=legend_element, loc="upper right", ncol=2)

    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
