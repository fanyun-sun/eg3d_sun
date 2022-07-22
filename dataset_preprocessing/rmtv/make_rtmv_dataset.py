import glob 
import subprocess 
import meshcat 
import simplejson as json 
from utils import * 
import numpy as np 
import cv2
from PIL import Image
import pyexr as exr
import os

dataset = {"labels":[]}
output = "data_google_shoe_single/"
subprocess.call(["mkdir",output])

def read_image_pillow(img_file):
    img = Image.open(img_file, 'r').convert('RGB')
    img = np.asarray(img).astype(np.float32)
    return img / 255.0

def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def read_image(file):
    if os.path.splitext(file)[1] == ".exr":
        img = exr.read(file).astype(np.float32)
    else:
        img = read_image_pillow(file)
        if img.shape[2] == 4:
            img[...,0:3] = srgb_to_linear(img[...,0:3])
            # Premultiply alpha
            img[...,0:3] *= img[...,3:4]
        else:
            img = srgb_to_linear(img)
    return img


path = "/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_centered/"
for scene in glob.glob(path + "*/"):
    folder_name = scene.replace(path,"")
    subprocess.call(["mkdir",output+folder_name])

    for i_pose, pose in enumerate(glob.glob(scene+"*.json")):
        with open(pose, 'r') as f:
            data = json.load(f)
        # print(data["camera_data"].keys())

        trans = np.array(data["camera_data"]['cam2world']).T    
        # trans[-1,0]*=10
        # trans[-1,1]*=10
        # trans[-1,2]*=10

        trans = visii_camera_frame_to_rdf(trans)
        trans = trans.flatten()


        intrinsics = data['camera_data']['intrinsics']
        intrinsics = [intrinsics['fx']/800,0,intrinsics['cx']/800,
                      0, intrinsics['fy']/800,intrinsics['cy']/800,
                      0,0,1  ]
        name = pose.replace(path,"").replace("json",'png')
        to_add = [name, trans.tolist() + intrinsics]
        # print(to_add)
        dataset['labels'].append(to_add)
        # raise()
        ref_img_srgb = read_image(pose.replace("json",'exr'))
        ref_img_srgb = linear_to_srgb(ref_img_srgb)

        ref_img_srgb[...,:3] += (1.0 - ref_img_srgb[...,3:4]) * 1.0
        ref_img_srgb= ref_img_srgb[:,:,:3]
        ref_img_srgb = cv2.cvtColor(ref_img_srgb, cv2.COLOR_BGR2RGB)
        ref_img_srgb[ref_img_srgb>1]=1
        print(name)
        cv2.imwrite(output + name,ref_img_srgb*255)
        # raise()
        # make a png file


with open(output+'dataset.json', 'w') as json_file:
    json.dump(dataset, json_file)

 