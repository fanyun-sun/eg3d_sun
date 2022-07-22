import os
import gdown
import shutil
import tempfile
import subprocess


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as working_dir:
        download_name = 'cars_train.zip'
        url = 'https://drive.google.com/uc?id=1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn'
        output_dataset_name = 'cars_256.zip'

        dir_path = os.path.dirname(os.path.realpath(__file__))
        extracted_data_path = os.path.join(working_dir, os.path.splitext(download_name)[0])

        print("Downloading data...")
        zipped_dataset = os.path.join(working_dir, download_name)
        gdown.download(url, zipped_dataset, quiet=False)

        print("Unzipping downloaded data...")
        shutil.unpack_archive(zipped_dataset, working_dir)

        print("Converting camera parameters...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_shapenet_cameras.py')} --source={extracted_data_path}"
        subprocess.run([cmd], shell=True)

        print("Creating dataset zip...")
        cmd = f"python {os.path.join(dir_path, '../../eg3d', 'dataset_tool.py')}"
        cmd += f" --source {extracted_data_path} --dest {output_dataset_name} --resolution 256x256"
        subprocess.run([cmd], shell=True)