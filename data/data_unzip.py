import os
import zipfile

"""
Unzips the images to a folder
"""


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
    save_directory = "./Downloads/Tiny_Portraits_Images/"  # change to appropriate folder to save the unziped output
    zip_directory = "./Downloads/TinyPortraits/"  # path to zip file
    unzip_data(zip_directory, save_directory)
