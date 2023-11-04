"""
1) Generate histograms for all resized as well as histogram equalized images
2) r_mean, r_std, g_mean, g_std, b_mean, b_std, rgb_mean, rgb_std
"""

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import io


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_histogram_data(source_image_path: str, histogram_output_path: str, image_type):
    img = cv2.imread(source_image_path, -1)
    data = {}
    color = ("b", "g", "r")
    f_t = 0
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        f = histr.flatten()
        if type(f_t) == int:
            f_t = f
        else:
            f_t = f_t + f

        x = np.array(range(0, 256))
        x_mean = np.sum(f * x) / np.sum(f)
        x_std = (np.sum(f * ((x - x_mean) ** 2)) / np.sum(f)) ** 0.5
        data[image_type + "_" + col + "_mean"] = x_mean
        data[image_type + "_" + col + "_std"] = x_std

        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    data[image_type + "_total_mean"] = np.sum(f_t * x) / np.sum(f_t)
    data[image_type + "_total_std"] = (
        np.sum(f_t * ((x - data[image_type + "_total_mean"]) ** 2)) / np.sum(f_t)
    ) ** 0.5

    plt.title("Histogram for color scale picture")

    fig = plt.gcf()
    histogram = fig2img(fig)
    histogram.save(histogram_output_path)
    plt.clf()

    return data


def get_dataframe_row(resized_image_path, equalized_image_path):
    data1 = get_histogram_data(
        resized_image_path,
        resized_image_path.replace("output", "histograms")
        .replace("jpg", "png")
        .replace("tif", "png"),
        "resized",
    )
    data2 = get_histogram_data(
        equalized_image_path,
        equalized_image_path.replace("output", "histograms")
        .replace("jpg", "png")
        .replace("tif", "png"),
        "equalized",
    )
    data1.update(data2)
    data1["resized_histogram_path"] = (
        resized_image_path.replace("output", "histograms")
        .replace("jpg", "png")
        .replace("tif", "png")
    )
    data1["equalized_histogram_path"] = (
        equalized_image_path.replace("output", "histograms")
        .replace("jpg", "png")
        .replace("tif", "png")
    )

    return data1


model_names = ["final_train_allidb1.pth", "final_train_allidb2.pth"]
image_types = ["resized", "equalized"]

if not os.path.exists("histograms"):
    os.makedirs("histograms")

for model_name in model_names:
    if not os.path.exists(os.path.join("histograms", model_name)):
        os.makedirs(os.path.join("histograms", model_name))
    for image_type in image_types:
        if not os.path.exists(os.path.join("histograms", model_name, image_type)):
            os.makedirs(os.path.join("histograms", model_name, image_type))

df1 = pd.read_csv(r"output\report_dataframe.csv")

df2 = pd.DataFrame(
    columns=[
        "resized_histogram_path",
        "resized_b_mean",
        "resized_b_std",
        "resized_g_mean",
        "resized_g_std",
        "resized_r_mean",
        "resized_r_std",
        "resized_total_mean",
        "resized_total_std",
        "equalized_histogram_path",
        "equalized_b_mean",
        "equalized_b_std",
        "equalized_g_mean",
        "equalized_g_std",
        "equalized_r_mean",
        "equalized_r_std",
        "equalized_total_mean",
        "equalized_total_std",
    ]
)

for idx, row in df1.iterrows():
    new_row = pd.DataFrame(
        get_dataframe_row(row["resized_path"], row["equalized_path"]), index=[0]
    )
    df2 = pd.concat([df2, new_row], ignore_index=True)
    print(idx)

df1 = df1.iloc[: len(df2), :]
df3 = pd.concat([df1, df2], axis=1)
df3.to_csv("complete_report_dataframe.csv", index=False)
