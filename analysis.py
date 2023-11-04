import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import io


def correct_prediction_distribution(start: np.ndarray, end: np.ndarray, n_images):
    a = np.arange(n_images * 256).reshape((256, n_images))
    mask = np.logical_and(
        np.arange(len(a))[:, None] >= start, np.arange(len(a))[:, None] <= end
    ).T
    return mask.sum(axis=0).astype(float)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_histogram_data(f, histogram_output_path):
    row = {}

    x = np.array(range(0, 256))

    row["mean"] = np.sum(f * x) / np.sum(f)
    row["std"] = (np.sum(f * ((x - row["mean"]) ** 2)) / np.sum(f)) ** 0.5

    plt.plot(f)
    plt.xlim([0, 256])

    fig = plt.gcf()
    histogram = fig2img(fig)
    histogram.save(histogram_output_path)
    plt.clf()

    return row


def correct_prediction_distribution(start: np.ndarray, end: np.ndarray, n_images):
    a = np.arange(n_images * 256).reshape((256, n_images))
    mask = np.logical_and(
        np.arange(len(a))[:, None] >= start, np.arange(len(a))[:, None] <= end
    ).T
    return mask.sum(axis=0).astype(float)


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_histogram_data(f, histogram_output_path):
    row = {}

    x = np.array(range(0, 256))

    row["mean"] = np.sum(f * x) / np.sum(f)
    row["std"] = (np.sum(f * ((x - row["mean"]) ** 2)) / np.sum(f)) ** 0.5

    plt.plot(f)
    plt.xlim([0, 256])

    fig = plt.gcf()
    histogram = fig2img(fig)
    histogram.save(histogram_output_path)
    plt.clf()

    return row


def main():
    df = pd.read_csv("complete_report_dataframe(wrong_total).csv", index_col=0)

    datasets = ["ALL_IDB1", "ALL_IDB2"]
    image_types = ["resized", "equalized"]
    types_of_noise = [
        "BLR1",
        "BLR10",
        "BLR11",
        "BLR12",
        "BLR2",
        "BLR3",
        "BLR4",
        "BLR5",
        "BLR6",
        "BLR7",
        "BLR8",
        "BLR9",
        "HB1",
        "HB10",
        "HB11",
        "HB12",
        "HB13",
        "HB2",
        "HB3",
        "HB4",
        "HB5",
        "HB6",
        "HB7",
        "HB8",
        "HB9",
        "HC1",
        "HC10",
        "HC2",
        "HC3",
        "HC4",
        "HC5",
        "HC6",
        "HC7",
        "HC8",
        "HC9",
        "LB1",
        "LB10",
        "LB11",
        "LB12",
        "LB13",
        "LB2",
        "LB3",
        "LB4",
        "LB5",
        "LB6",
        "LB7",
        "LB8",
        "LB9",
        "LC1",
        "LC10",
        "LC2",
        "LC3",
        "LC4",
        "LC5",
        "LC6",
        "LC7",
        "LC9",
        "LC8",
    ]

    data = {}

    if not os.path.exists("analyzed_output"):
        os.makedirs("analyzed_output")

    output_df = pd.DataFrame(
        columns=[
            "dataset",
            "image_type",
            "noise_type",
            "confusion_matrix_category",
            "channel",
            "mean",
            "std",
            "count",
        ]
    )

    for dataset in datasets:
        for noise_type in types_of_noise:
            for image_type in image_types:
                channels = ["r", "g", "b", "total"]
                for channel in channels:
                    df_channel = df[
                        (df["dataset"] == dataset) & (df["noise_type"] == noise_type)
                    ].copy()

                    df_channel["start"] = (
                        df[image_type + "_" + channel + "_mean"]
                        - df[image_type + "_" + channel + "_std"]
                    )
                    df_channel["end"] = (
                        df[image_type + "_" + channel + "_mean"]
                        + df[image_type + "_" + channel + "_std"]
                    )

                    confusion_matrix_category = {
                        "tp": df_channel.query(
                            "predicted_label == 1 & actual_label == 1"
                        ),
                        "tn": df_channel.query(
                            "predicted_label == 0 & actual_label == 0"
                        ),
                        "fp": df_channel.query(
                            "predicted_label == 1 & actual_label == 0"
                        ),
                        "fn": df_channel.query(
                            "predicted_label == 0 & actual_label == 1"
                        ),
                    }

                    for category in confusion_matrix_category:
                        row = {}
                        if len(confusion_matrix_category[category]) == 0:
                            row["mean"] = 0
                            row["std"] = 0
                        else:
                            mask = correct_prediction_distribution(
                                confusion_matrix_category[category]["start"].values,
                                confusion_matrix_category[category]["end"].values,
                                len(confusion_matrix_category[category]),
                            )
                            row = get_histogram_data(
                                mask,
                                f"analyzed_output/{dataset}_{image_type}_{noise_type}_{category}_{channel}.png",
                            )

                        row["dataset"] = dataset
                        row["image_type"] = image_type
                        row["noise_type"] = noise_type
                        row["confusion_matrix_category"] = category
                        row["channel"] = channel
                        row["count"] = len(confusion_matrix_category[category])

                        output_df = pd.concat([output_df, pd.DataFrame(row, index=[0])])

    output_df.to_csv("analyzed_output.csv")


main()
