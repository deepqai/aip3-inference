# encoding=utf-8

import json
import os
from functools import partial

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.tableofcontents import TableOfContents

from .plot import (
    horizontal_bar_chart,
    plot_case_studies,
    plot_confusion_matrix,
    plot_model_performance,
)


class MCLine(Flowable):
    def __init__(self, width, height=0):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Line(w={self.width})"

    def draw(self):
        self.canv.setFillColorRGB(17 / 255, 69 / 255, 108 / 255)
        self.canv.line(0, self.height, self.width, self.height)


class InsightReport(BaseDocTemplate):
    def __init__(
        self,
        filename,
        task_type,
        data_insight_dir,
        insight_content_json,
        report_setting_json,
        datadir=None,
        class_list=None,
        train_realtime_statistics=None,
        inference_statistics=None,
        case_study_examples=None,
        **kw,
    ):
        super().__init__(filename, **kw)

        self.task_type = task_type
        self.data_insight_dir = data_insight_dir
        self.cover_name = filename.split("/")[-1][:-4].replace("_", " ").title()

        with open(
            os.path.join(os.path.dirname(__file__), insight_content_json), "r"
        ) as f:
            self.insight_content_config = json.load(f)[self.task_type]

        with open(
            os.path.join(os.path.dirname(__file__), report_setting_json), "r"
        ) as f:
            self.report_config = json.load(f)

        self.datadir = datadir
        self.class_list = (
            class_list[1:] if self.task_type == "detection" else class_list
        )
        self.model_performance_dir = os.path.join(
            self.data_insight_dir, "model_performance"
        )
        self.case_study_plot_dir = os.path.join(
            self.data_insight_dir, "case_study_examples"
        )

        self.train_realtime_statistics = train_realtime_statistics
        self.inference_statistics = inference_statistics
        self.case_study_examples = case_study_examples

        self.skip_first_n_pages = 2

        for font_name, font_path in self.report_config["font"].items():
            pdfmetrics.registerFont(
                TTFont(font_name, os.path.join(os.path.dirname(__file__), font_path))
            )

        self.content_indent_level = {}
        for indent_name, setting in self.report_config["content_indent_level"].items():
            self.content_indent_level[indent_name] = PS(name=indent_name, **setting)

        self.toc_indent_level = {}
        for indent_name, setting in self.report_config["toc_indent_level"].items():
            self.toc_indent_level[indent_name] = PS(name=indent_name, **setting)

        cover_page = PageTemplate(
            "CoverPage",
            [Frame(A4[0] // 4, A4[1] // 4, 100, 100, id="cover")],
            onPage=self.generate_cover_page,
        )
        index_page = PageTemplate(
            "IndexPage",
            frames=[Frame(20, 40, 550, 750, id="index")],
            onPage=partial(
                self.format_page,
                use_header=True,
                use_watermark=False,
                use_footer=False,
            ),
        )
        normal_page = PageTemplate(
            "NormalPage",
            frames=[Frame(20, 40, 550, 750, id="normal")],
            onPage=partial(
                self.format_page,
                use_header=True,
                use_watermark=True,
                use_footer=True,
            ),
        )

        self.addPageTemplates(
            [
                cover_page,
                index_page,
                normal_page,
            ]
        )

        self.page_handlers = {
            "generate_data_statistics_figures": self.generate_data_statistics_figures,
            "generate_data_statistics_tables": self.generate_data_statistics_tables,
            "generate_model_performance_figures": self.generate_model_performance_figures,
            "generate_model_performance_table": self.generate_model_performance_table,
            "generate_case_studies": self.generate_case_studies,
        }

        self.story = []

    def afterFlowable(self, flowable):
        "Registers TOC entries."
        if isinstance(flowable, Paragraph):
            text = flowable.getPlainText()

            if text == "Index":
                self.canv.bookmarkPage(text)
                self.canv.addOutlineEntry(text, text, 0, 0)
                return

            style = flowable.style.name
            if style == "Heading1":
                level = 0
            elif style == "Heading2":
                level = 1
            else:
                return

            if text.startswith("Label:") or text == "Case Studies":
                key = text
            else:
                key = f"{self.canv.getPageNumber()}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level, 0)
            E = [level, text, self.page - self.skip_first_n_pages, key]
            self.notify("TOCEntry", E)

    def format_page(self, canvas, doc, use_header, use_watermark, use_footer):
        canvas.saveState()

        if use_header:
            self.header(canvas)

        if use_watermark:
            self.watermark(canvas)

        if use_footer:
            self.footer(canvas)

        canvas.restoreState()

    def header(self, canvas):
        canvas.drawImage(
            os.path.join(
                os.path.dirname(__file__), "./supporting_files/deepq-logo.png"
            ),
            A4[0] - 110,
            A4[1] - 50,
            width=50,
            height=23,
            preserveAspectRatio=True,
            mask="auto",
        )

    def watermark(self, canvas):
        canvas.drawImage(
            os.path.join(
                os.path.dirname(__file__), "./supporting_files/deepq_watermark.png"
            ),
            x=A4[0] // 32,
            y=A4[1] // 10,
            width=550,
            height=750,
            preserveAspectRatio=True,
            mask="auto",
        )

    def footer(self, canvas):
        canvas.setFont("GenShinGothic", 8)
        cur_page = canvas.getPageNumber()
        canvas.drawCentredString(
            A4[0] // 2,
            20,
            f"{self.cover_name} - pg.{cur_page - self.skip_first_n_pages}",
        )

        canvas.setFont("roboto_I", 8)
        canvas.setFillColorRGB(63 / 255, 130 / 255, 210 / 255)
        canvas.drawString(A4[0] - 100, 20, "click to return to index")
        canvas.linkAbsolute(
            "(click to return to index)",
            "Index",
            Rect=(A4[0] - 100, 20, A4[0] - 20, 28),
        )

    def generate_cover_page(self, canvas, doc):
        canvas.saveState()

        canvas.drawImage(
            os.path.join(
                os.path.dirname(__file__), "./supporting_files/cover_background.png"
            ),
            0,
            0,
            width=A4[0],
            height=A4[1],
        )

        canvas.setFont("GenShinGothic_B", 34)
        canvas.setFillColorRGB(1, 1, 1)
        canvas.drawString(30, A4[1] // 2, self.cover_name)

        canvas.drawImage(
            os.path.join(os.path.dirname(__file__), "./supporting_files/aip_logo.png"),
            30,
            A4[1] - 50,
            width=110,
            height=28,
            preserveAspectRatio=True,
            mask="auto",
        )

        canvas.restoreState()

    def generate_index_page(self):
        self.story.append(PageBreak())

        self.story.append(Paragraph("Index", self.content_indent_level["Heading1"]))
        self.story.append(Spacer(0, 18))

        self.story.append(MCLine(540))
        self.story.append(Spacer(0, 18))

        toc = TableOfContents()
        toc.levelStyles = [
            self.toc_indent_level["TOCHeading1"],
            self.toc_indent_level["TOCHeading2"],
        ]
        self.story.append(toc)

    def generate_content_page(self):
        self.story.append(PageBreak())

        for heading1, heading1_info in self.insight_content_config.items():
            if not heading1_info["content_info"]:
                continue

            self.story.append(
                Paragraph(heading1, self.content_indent_level["Heading1"])
            )
            self.story.append(Spacer(0, 18))

            self.story.append(MCLine(540))
            self.story.append(Spacer(0, 18))

            if heading1_info.get("note", None):
                self.story.append(
                    Paragraph(
                        heading1_info["note"], self.content_indent_level["Heading1Note"]
                    )
                )
                self.story.append(Spacer(0, 18))

            handler = self.page_handlers[heading1_info["handler"]]
            handler(heading1_info["content_info"])

    def generate_data_statistics_figures(self, content_info):
        for header, info in content_info.items():

            dataset_folder = info["dataset_folder"]
            dataset_name = info["dataset_name"]

            self.story.append(
                Paragraph(
                    header,
                    self.content_indent_level["Heading2"],
                )
            )
            self.story.append(Spacer(0, 18))

            self.story.append(
                Paragraph(
                    info["description"],
                    self.content_indent_level["Heading2Description"],
                )
            )
            self.story.append(Spacer(0, 18))

            import pdb;pdb.set_trace()
            for class_name in ["all_labels"] + self.class_list:
                each_content = []
                if not os.path.exists(
                    os.path.join(
                        self.data_insight_dir,
                        dataset_folder[0],
                        "csv",
                        f"{info['body']}_{class_name}.csv",
                    )
                ):
                    continue

                table_data = [dataset_name]

                data = []
                y_limit = 0
                for folder in dataset_folder:
                    df = pd.read_csv(
                        os.path.join(
                            self.data_insight_dir,
                            folder,
                            "csv",
                            f"{info['body']}_{class_name}.csv",
                        ),
                    )
                    y_limit = max(y_limit, df.iloc[:-1, -1].max())
                    data.append(df)

                row_data = []
                for df, folder in zip(data, dataset_folder):
                    save_path = os.path.join(
                        self.data_insight_dir,
                        folder,
                        "img",
                        f"{info['body']}_{class_name}.png",
                    )
                    process_df = df.iloc[:-1, [0, -1]]

                    stat_dict = process_df.set_index("name").to_dict()["percentage"]
                    horizontal_bar_chart(
                        stat_dict, save_path, y_limit, info["show_full_len"]
                    )
                    row_data.append(
                        Image(
                            save_path,
                            275,
                            A4[1],
                            kind="proportional",
                            mask="auto",
                        )
                    )
                table_data.append(row_data)

                overall_table = Table(table_data, colWidths=[275, 275])
                overall_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (1, 1), "CENTER"),
                        ]
                    )
                )

                class_name = class_name if class_name != "all_labels" else "all"

                each_content.append(
                    Paragraph(
                        f"Label: {class_name}",
                        self.content_indent_level["ContentIntro"],
                    )
                )
                each_content.append(Spacer(0, 18))
                each_content.append(overall_table)
                each_content.append(Spacer(0, 18))

                self.story.append(KeepTogether(each_content))

            self.story.append(PageBreak())

    def generate_data_statistics_tables(self, content_info):
        for header, info in content_info.items():
            dataset_folder = info["dataset_folder"]
            dataset_name = info["dataset_name"]

            self.story.append(
                Paragraph(
                    header,
                    self.content_indent_level["Heading2"],
                )
            )
            self.story.append(Spacer(0, 18))

            self.story.append(
                Paragraph(
                    info["description"],
                    self.content_indent_level["Heading2Description"],
                )
            )
            self.story.append(Spacer(0, 18))

            for class_name in ["all_labels"] + self.class_list:
                each_content = []
                if not os.path.exists(
                    os.path.join(
                        self.data_insight_dir,
                        dataset_folder[0],
                        "csv",
                        f"{info['body']}_{class_name}.csv",
                    )
                ):
                    continue

                dataset_df = []
                for folder in dataset_folder:
                    df = pd.read_csv(
                        os.path.join(
                            self.data_insight_dir,
                            folder,
                            "csv",
                            f"{info['body']}_{class_name}.csv",
                        )
                    )
                    df.loc[:, "name"] = df.loc[:, "name"].apply(
                        lambda x: x if len(x) <= 16 else f"{x[:16]}..."
                    )
                    df.loc[:, "percentage"] = df.loc[:, "percentage"].apply(
                        lambda x: f"{x:.2f}"
                    )
                    dataset_df.append(df)

                if info["body"] in ["multi_label", "img_wise", "label_distribution"]:
                    dataset_total_num = [df.iloc[-1, 1].copy() for df in dataset_df]
                    dataset_df = [df.iloc[:-1, :] for df in dataset_df]

                dataset_table = []
                for df in dataset_df:
                    df_table = Table(
                        [["Label", f"# of {info['count_base']}", "Percentage"]]
                        + df.values.tolist(),
                    )
                    df_table.setStyle(
                        TableStyle(
                            [
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("FONT", (0, 0), (-1, 0), "roboto_B"),
                                ("FONT", (0, 1), (0, -1), "GenShinGothic"),
                            ]
                        )
                    )
                    dataset_table.append(df_table)

                overall_table_data = [
                    dataset_name,
                    dataset_table,
                ]

                if info["body"] in ["multi_label", "img_wise", "label_distribution"]:
                    overall_table_data.append(
                        [
                            f"TOTAL NUMBER OF IMAGES: {total_num}"
                            for total_num in dataset_total_num
                        ]
                    )

                overall_table = Table(overall_table_data, colWidths=[275, 275])
                overall_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ]
                    )
                )

                class_name = class_name if class_name != "all_labels" else "all"
                each_content.append(
                    Paragraph(
                        f"Label: {class_name}",
                        self.content_indent_level["ContentIntro"],
                    )
                )
                each_content.append(Spacer(0, 18))
                each_content.append(overall_table)
                each_content.append(Spacer(0, 18))
                self.story.append(KeepTogether(each_content))

            self.story.append(PageBreak())

    def generate_model_performance_figures(self, content_info):
        for header, info in content_info.items():

            self.story.append(
                Paragraph(
                    header,
                    self.content_indent_level["Heading2"],
                )
            )
            self.story.append(Spacer(0, 18))

            each_content = []
            for folder, name in zip(info["dataset_folder"], info["dataset_name"]):
                if header == "Confusion Matrix":
                    plot_confusion_matrix(
                        np.array(self.inference_statistics[folder]["confusion_matrix"]),
                        self.class_list,
                        save_path=os.path.join(
                            self.model_performance_dir, folder, f"{info['body']}.png"
                        ),
                    )
                else:
                    metric_x, metric_y = info["metric_x"], info["metric_y"]
                    axis_name = (info["metric_x_name"], info["metric_y_name"])

                    average_metric_val = "0"
                    plot_data = {}
                    for class_statistics in self.inference_statistics[folder].get(
                        "labels", []
                    ):
                        if class_statistics["name"] == "average":
                            average_metric_val = (
                                f"{class_statistics[info['average_metric']]:.3f}"
                            )
                            continue

                        if class_statistics[metric_x] and class_statistics[metric_y]:
                            legend_description = ""
                            if self.task_type != "detection":
                                legend_description = f" ({info['class_wise_average_metric_name']}={class_statistics[info['average_metric']]:.3f}%)"

                            plot_data[class_statistics["name"]] = (
                                class_statistics[metric_x],
                                class_statistics[metric_y],
                                legend_description,
                            )

                    if self.task_type == "detection":
                        average_metric_val = "50"

                    plot_model_performance(
                        plot_data=plot_data,
                        axis_name=axis_name,
                        average_performance=(
                            info["average_metric_name"],
                            average_metric_val,
                        ),
                        dash_line=info.get("dash_line", None),
                        save_path=os.path.join(
                            self.model_performance_dir, folder, f"{info['body']}.png"
                        ),
                    )

                body = Image(
                    os.path.join(
                        self.model_performance_dir,
                        folder,
                        f"{info['body']}.png",
                    ),
                    450,
                    A4[1],
                    kind="proportional",
                    mask="auto",
                )

                each_content.append(
                    Paragraph(
                        name,
                        self.content_indent_level["ContentIntro"],
                    )
                )
                each_content.append(Spacer(0, 18))
                each_content.append(body)
                each_content.append(Spacer(0, 18))
                self.story.append(KeepTogether(each_content))

            self.story.append(PageBreak())

    def generate_model_performance_table(self, content_info):
        for header, info in content_info.items():
            self.story.append(
                Paragraph(
                    header,
                    self.content_indent_level["Heading2"],
                )
            )
            self.story.append(Spacer(0, 18))

            each_content = []
            for folder, name in zip(info["dataset_folder"], info["dataset_name"]):

                table_data = [[info["row_name"]] + list(info["column_names"].keys())]
                content_key = "summary" if self.task_type == "detection" else "labels"
                for class_statistics in self.inference_statistics[folder][content_key]:
                    row_data = [class_statistics["name"]]
                    for k, v in info["column_names"].items():
                        metric = class_statistics.get(v["metric"], "-")
                        if metric != "-" and v["unit"] == "percentage":
                            metric = f"{metric:.3f}"
                        else:
                            metric = str(metric)
                        row_data.append(metric)
                    table_data.append(row_data)

                column_width = [
                    self.report_config["table_column_width"]["one_column"]
                ] * len(table_data[0])

                table = Table(table_data, colWidths=column_width)
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("FONT", (0, 0), (-1, 0), "roboto_B"),
                            ("FONT", (0, 1), (0, -1), "GenShinGothic"),
                        ]
                    )
                )
                each_content.append(
                    Paragraph(
                        name,
                        self.content_indent_level["ContentIntro"],
                    )
                )
                each_content.append(Spacer(0, 18))
                each_content.append(table)
                each_content.append(Spacer(0, 18))
                self.story.append(KeepTogether(each_content))

            self.story.append(
                Paragraph(
                    info["content_note"],
                    self.content_indent_level["ContentNote"],
                )
            )

            self.story.append(PageBreak())

    def generate_case_studies(self, content_info):
        for class_name in self.class_list:
            body = Paragraph(
                f"<a href='#Label: {class_name}' color='blue'>{class_name}</a>",
                self.content_indent_level["Heading2Description"],
            )
            self.story.append(body)
        self.story.append(PageBreak())

        for class_name in self.class_list:

            self.story.append(
                Paragraph(
                    f"Label: {class_name}",
                    self.content_indent_level["Heading2"],
                )
            )
            self.story.append(Spacer(0, 18))

            case_studies = self.case_study_examples[class_name]
            for case, full_name in {
                "TP": "True Positive",
                "FP": "False Positive",
                "FN": "False Negative",
            }.items():
                each_content = []
                case_study_plot_path = os.path.join(
                    self.case_study_plot_dir, class_name, f"{case}.png"
                )
                plot_case_studies(
                    task_type=self.task_type,
                    plot_data=case_studies.get(case, []),
                    n_row=3,
                    n_col=6,
                    datadir=self.datadir,
                    save_path=case_study_plot_path,
                )
                body = Image(
                    case_study_plot_path,
                    380,
                    A4[1],
                    kind="proportional",
                    mask="auto",
                )
                each_content.append(
                    Paragraph(
                        full_name,
                        self.content_indent_level["ContentIntro"],
                    )
                )
                each_content.append(Spacer(0, 18))
                each_content.append(body)
                each_content.append(Spacer(0, 18))
                self.story.append(KeepTogether(each_content))

            self.story.append(
                Paragraph(
                    "<a href='#Case Studies' color='blue'>Return back to Case Studies Index</a>",
                    PS(name="CenterLink", alignment=TA_CENTER),
                )
            )

            self.story.append(PageBreak())

    def run(self):
        self.story.append(NextPageTemplate("IndexPage"))
        self.generate_index_page()

        self.story.append(NextPageTemplate("NormalPage"))
        self.generate_content_page()

        self.multiBuild(self.story)
