import os
import zipfile
import pandas as pd
from numpy.random import RandomState

"""
Unzips the images to a folder
"""


def split_data(save_directory, csv_path):
    df = pd.read_csv(csv_path)

    # checks if the index in csv have a corresponding image in the main_dir. If not drop the index
    indices = [
        os.path.exists(
            os.path.join(
                save_directory, "Tiny_Portrait_{:06d}.png".format(df.iloc[index, 0])
            )
        )
        for index in range(len(df))
    ]
    df_clean = df[indices]

    # splits the csv into two for training and testing
    rng = RandomState()
    train = df_clean.sample(frac=0.9, random_state=rng)
    test = df_clean.loc[~df_clean.index.isin(train.index)]

    save_dir = os.path.dirname(csv_path)
    train.to_csv(
        os.path.join(save_dir, "Tiny_Portraits_Attributes_Train.csv"), index=False
    )
    test.to_csv(
        os.path.join(save_dir, "Tiny_Portraits_Attributes_Test.csv"), index=False
    )
    print("Number of Samples in Training set: %d" % (len(train)))
    print("Number of Samples in Test set: %d" % (len(test)))


def unzip_data(zip_directory, save_directory):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load ordered list of archives
    archive_file_list = [
        file
        for file in sorted(os.listdir(zip_directory))
        if file.split(".")[-1] == "zip"
    ]
    archive_file_count = len(archive_file_list)

    # Check for completeness
    assert archive_file_count == 66

    for index in range(0, archive_file_count):
        # Verify archive name
        zip_archive_name = "Tiny_Portraits_Archive_{:03d}.zip".format(index)
        assert archive_file_list[index] == zip_archive_name

        # Extract image files
        with zipfile.ZipFile(zip_directory + zip_archive_name, "r") as archive:
            archive.extractall(zip_directory)

        # Indicate progress
        num_images = len(
            [file for file in os.listdir(zip_directory) if file.split(".")[-1] == "png"]
        )
        print(
            "\rUnzipping archive #{:3d} ... Total images # {:6d}".format(
                index, num_images
            ),
            end="",
        )

    # Verify number of images
    assert num_images == 134734

    print("\n*** DONE ***")


if __name__ == "__main__":
    zip_directory = "./Downloads/TinyPortraits/"  # path to zip file
    save_directory = "/Users/nimisha/Downloads/Tiny_Portraits_Images/"
    csv_path = "debiased_classifier/data/Tiny_Portraits_Attributes.csv"

    if os.path.exists(save_directory) and any(os.scandir(save_directory)):
        split_data(save_directory, csv_path)
    else:
        unzip_data(zip_directory, save_directory)
        split_data(save_directory, csv_path)
