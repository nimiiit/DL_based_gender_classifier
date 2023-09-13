import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from skimage import io
from sklearn.preprocessing import LabelEncoder
import os


def general_transforms(img_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def augmentations(img_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.ToTensor(),
        ]
    )


def create_dataset(config, mode="train"):
    """
    Use the config file to create data loader
    Splits the training set to (0.8,0.2) ratio for training and validation
    """
    selected_attrib = config["selected_attribute"]
    img_size = config["image_size"]
    transforms = general_transforms(img_size)
    datasets = TinyPortrait(
        config["data_path"], config["csv_path"], selected_attrib, transform=transforms
    )

    class_name = datasets.cls_names()

    if mode == "train":
        # define the split for train, val
        train_size = int(0.8 * len(datasets))
        val_size = len(datasets) - train_size
        train_set, val_set = torch.utils.data.random_split(
            datasets, [train_size, val_size]
        )
        aug_transform = augmentations(img_size)
        train_set = augment(train_set, aug_transform)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config["batch_size"], num_workers=2
        )
        return [train_loader, val_loader, class_name]
    elif mode == "test":
        test_loader = torch.utils.data.DataLoader(
            datasets, batch_size=config["batch_size"], num_workers=2
        )
        return [test_loader, class_name]


class TinyPortrait(Dataset):
    def __init__(self, main_dir, csv_file, selected_attrib=None, transform=None):
        df = pd.read_csv(csv_file)[["Image_Index", selected_attrib]]
        df.dropna(axis=0, inplace=True)
        attrib_list = df.select_dtypes(include="object").columns
        lab_encoder = LabelEncoder()
        for feat in attrib_list:
            df[feat] = lab_encoder.fit_transform(df[feat].astype(str))
            label_name_mapping = dict(
                zip(lab_encoder.classes_, lab_encoder.transform(lab_encoder.classes_))
            )

        self.class_names = label_name_mapping
        self.label_map = df
        self.main_dir = main_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_map)

    def cls_names(self):
        return self.class_names

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.main_dir,
            "Tiny_Portrait_{:06d}.png".format(self.label_map.iloc[idx, 0]),
        )
        image = io.imread(image_path)

        # provide sample weights conditioned on gender for unbiasing
        label = torch.tensor((self.label_map.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, label)


class augment(Dataset):
    """
    Performs augmentations on a dataset
    Used in our framework for augmenting the training set alone
    """

    def __init__(self, datasets, transform):
        self.datasets = datasets
        self.transform = transform

    def __getitem__(self, idx):
        im, labels, weights = self.datasets[idx]
        return (self.transform(im), labels, weights)

    def __len__(self):
        return len(self.datasets)
