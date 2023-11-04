import os
import cv2
import torch
import torchvision
import pandas as pd
from torchvision.models import mobilenet_v2
from torchvision.utils import save_image


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mbv2 = mobilenet_v2(pretrained=True)
        self.mbv2.classifier[1] = torch.nn.Linear(
            self.mbv2.classifier[1].in_features, 1
        )
        torch.set_grad_enabled(True)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.mbv2(x)
        x = self.sigm(x)
        return x


def main():
    model_names = ["final_train_allidb1.pth", "final_train_allidb2.pth"]
    dataset_names = ["ALL_IDB1", "ALL_IDB2"]
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
    df = pd.DataFrame(
        columns=[
            "dataset",
            "souurce_image_path",
            "resized_path",
            "equalized_path",
            "noise_type",
            "predicted_label",
            "actual_label",
        ]
    )

    transform1 = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 512)),
        ]
    )
    transform2 = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomEqualize(1),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("output"):
        os.makedirs("output")

    for model_name in model_names:
        if not os.path.exists(f"output/{model_name}"):
            os.makedirs(f"output/{model_name}")

        if not os.path.exists(f"output/{model_name}/resized"):
            os.makedirs(f"output/{model_name}/resized")

        if not os.path.exists(f"output/{model_name}/equalized"):
            os.makedirs(f"output/{model_name}/equalized")
    try:
        for model_name, dataset_name in zip(model_names, dataset_names):
            net = Model()
            saved_model = torch.load(os.path.join("saved_models", model_name))
            net.load_state_dict(saved_model["net"])
            net = net.to(device)

            net.eval()

            dataset_total = 0
            dataset_correct = 0
            for type_of_noise in types_of_noise:
                noise_total = 0
                noise_correct = 0

                images_in_folder = os.listdir(
                    f"data/{dataset_name}/Noise/{type_of_noise}"
                )

                for image_name in images_in_folder:
                    source_image_path = (
                        f"data/{dataset_name}/Noise/{type_of_noise}/{image_name}"
                    )
                    img = cv2.imread(source_image_path)
                    label = source_image_path.split("_")[-1][0]

                    img_tensor = torch.from_numpy(img).float()
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

                    img_tensor = transform1(img_tensor).float()

                    img_tensor = torch.as_tensor(img_tensor, dtype=torch.uint8)

                    resized_path = f"output/{model_name}/resized/{dataset_name}_{type_of_noise}_{image_name}"
                    save_image(img_tensor / 255, resized_path)

                    img_tensor = transform2(img_tensor)

                    img_tensor = torch.as_tensor(img_tensor, dtype=torch.float32)
                    equalized_path = f"output/{model_name}/equalized/{dataset_name}_{type_of_noise}_{image_name}"
                    save_image(img_tensor / 255, equalized_path)

                    img_tensor = img_tensor
                    label = torch.tensor(int(label))

                    #################################################

                    img_tensor = img_tensor.to(device)
                    label = label.to(device)

                    output = net(img_tensor)
                    output = output.squeeze(0)
                    predicted = int(output.ge(0.5).item())

                    if predicted == label.item():
                        noise_correct += 1
                        dataset_correct += 1
                    noise_total += 1
                    dataset_total += 1

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "dataset": dataset_name,
                                    "source_image_path": source_image_path,
                                    "resized_path": resized_path,
                                    "equalized_path": equalized_path,
                                    "noise_type": type_of_noise,
                                    "predicted_label": predicted,
                                    "actual_label": label.item(),
                                },
                                index=[0],
                            ),
                        ],
                        ignore_index=True,
                    )

                print(
                    f"{dataset_name} {type_of_noise} accuracy: {noise_correct}/{noise_total} = {noise_correct/noise_total * 100}%"
                )

            print(
                f"{dataset_name} accuracy: {dataset_correct}/{dataset_total} = {dataset_correct/dataset_total * 100}%"
            )
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt")
    df.to_csv("output/report_dataframe.csv", index=False)


main()
