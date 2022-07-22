#############################################################

# Usage: python dataset_preprocessing/metfaces/preprocess_metfaces_cameras.py --source /data/metfaces --dest /data/preprocessed_metfaces_images

#############################################################

import json
import numpy as np
from PIL import Image, ImageOps
import os
from tqdm import tqdm
import argparse
import shutil
    
def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

def fix_pose(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3]/radius * 2.7
    return pose

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

def center_crop_image(img, width, height):
    img = img.crop(box=((img.size[1] - width) // 2, (img.size[0] - height) // 2, (img.size[1] + width) // 2, (img.size[0] + height) // 2))
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    camera_dataset_file = os.path.join(args.source, 'cameras.json')
    if not os.path.isfile(camera_dataset_file):
        print("Copying cameras file")
        shutil.copyfile('dataset_preprocessing/metfaces/cameras.json', camera_dataset_file)

    with open(camera_dataset_file, "r") as f:
        cameras = json.load(f)
        
    dataset = {'labels':[]}

    max_images = args.max_images if args.max_images is not None else len(cameras)
    for i, filename in tqdm(enumerate(cameras), total=max_images):
        if (max_images is not None and i >= max_images): break

        pose = cameras[filename]['pose']
        intrinsics = cameras[filename]['intrinsics']
        pose = fix_pose(pose)
        intrinsics = fix_intrinsics(intrinsics)
        label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
            
        image_path = os.path.join(args.source, filename)
        img = Image.open(image_path)

        dataset["labels"].append([filename, label])
        os.makedirs(os.path.dirname(os.path.join(args.dest, filename)), exist_ok=True)
        img.save(os.path.join(args.dest, filename))


        flipped_img = ImageOps.mirror(img)
        flipped_pose = flip_yaw(pose)
        label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()
        base, ext = filename.split('.')[0], '.' + filename.split('.')[1]
        flipped_filename = base + '_mirror' + ext
        dataset["labels"].append([flipped_filename, label])
        flipped_img.save(os.path.join(args.dest, flipped_filename))
        
    with open(os.path.join(args.dest, 'dataset.json'), "w") as f:
        json.dump(dataset, f)