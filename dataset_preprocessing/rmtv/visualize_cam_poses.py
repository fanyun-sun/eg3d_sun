import glob 
import subprocess 
import meshcat 
import simplejson as json 
from utils import * 
import numpy as np 

vis = create_visualizer(True)


for scene in glob.glob("/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/data/google_shoe_single_centered/*/"):
    for i_pose, pose in enumerate(glob.glob(scene+"*.json")):
        with open(pose, 'r') as f:
            data = json.load(f)
        print(data["camera_data"].keys())
        trans = np.array(data["camera_data"]['cam2world']).T    
        # trans[-1,0]*=10
        # trans[-1,1]*=10
        # trans[-1,2]*=10

        trans = visii_camera_frame_to_rdf(trans)

        print(trans)

        make_frame(vis,str(i_pose)+"d",transform=trans)
    break
with open('dataset_preprocessing/shapenet_cars/dataset.json', 'r') as f:
    data = json.load(f)


for i, image in enumerate(data['labels']):
    img_folder = image[0].split('/')[0]
    if not img_folder == "00000":
        break
    trans = np.array(image[1][:16]).reshape(4,4)
    # print(trans)
    make_frame(vis,str(i),transform=trans)


