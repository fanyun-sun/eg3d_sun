import os
import sys
import shutil
import tempfile
import subprocess

import gdown

eg3d_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

download_name = 'afhq.zip'
output_dataset_name = 'afhq.zip'

dataset_tool_path = os.path.join(eg3d_root, 'eg3d', 'dataset_tool.py')
mirror_tool_path = os.path.join(eg3d_root, 'dataset_preprocessing', 'mirror_dataset.py')

# Attempt to import dataset_tool.py and mirror_dataset.py to fail-fast on errors (ie importing python modules) before any processing
try:
    sys.path.append(os.path.dirname(dataset_tool_path))
    import dataset_tool
    sys.path.append(os.path.dirname(mirror_tool_path))
    import mirror_dataset
except Exception as e:
    print(e)
    print("There was a problem while importing the dataset_tool. Are you in the correct virtual environment?")
    exit()


with tempfile.TemporaryDirectory() as working_dir:
    cmd = f"""
        URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0;
        ZIP_FILE={working_dir}/'afhq_v2.zip';
        wget -N $URL -O $ZIP_FILE;
        zip -FF $ZIP_FILE --out {working_dir}/repaired.zip;
        unzip {working_dir}/repaired.zip -d {working_dir}/extracted_images;
        mv {working_dir}/extracted_images/train/cat/ {working_dir}/cat_images/;
    """
    subprocess.run([cmd], shell=True)


    """Download dataset.json file"""
    json_url = 'https://drive.google.com/file/d/1FQXQ26kAgRyN2iOH8CBl3P9CGPIQ5TAQ/view?usp=sharing'
    gdown.download(json_url, f'{working_dir}/cat_images/dataset.json', quiet=False, fuzzy=True)


    print("Mirroring dataset...")
    cmd = f"""
        python {mirror_tool_path} \
            --source={working_dir}/cat_images \
            --dest={working_dir}/mirrored_images
    """
    subprocess.run([cmd], shell=True)


    print("Creating dataset zip...")
    cmd = f"""
        python {dataset_tool_path} \
            --source {working_dir}/mirrored_images \
            --dest {output_dataset_name} \
            --resolution 512x512
    """
    subprocess.run([cmd], shell=True)