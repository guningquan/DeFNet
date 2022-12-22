import torch
import torch.utils.data as data
import random
import pickle
import gc
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
import numpy as np
import os
import pickle
import math

def creat_dataset_rgbd_resize_actionlabel(data_path):
    """
    output:

    256*256*4
    """
    all_list = []
    trainfs = sorted(['_'.join(fn.split('_')[0:1])
                      for fn in os.listdir(f'{data_path}/actions')])
    for i in range(len(trainfs)):
        idx = trainfs[i]

        depth_before = np.load(f'{data_path}/rendered_images/{idx}_depth_before.npy')
        depth_before = cv2.resize(depth_before , (256, 256))  # the value can be 3000 or 200 or 100

        rgb_before = cv2.resize(np.array(Image.open(f'{data_path}/images/{idx}_rgb_before.png')), (256, 256))/255
        rgbd_before = np.concatenate((rgb_before, depth_before[:, :, None]), axis=-1)

        depth_after = np.load(f'{data_path}/rendered_images/{idx}_depth_after.npy')
        depth_after = cv2.resize(depth_after, (256, 256))  # the value can be 3000 or 200 or 100

        rgb_after = cv2.resize(np.array(Image.open(f'{data_path}/images/{idx}_rgb_after.png')), (256, 256))/255
        rgbd_after = np.concatenate((rgb_after, depth_after[:, :, None]), axis=-1)

        before_knot = np.load(f'{data_path}/knots/{idx}_knots_before.npy')
        after_knot = np.load(f'{data_path}/knots/{idx}_knots_after.npy')

        tmp = 0

        for k in range(before_knot.shape[0]):
            dist = np.hypot(*(before_knot[k] - after_knot[k]))
            if tmp < dist:
                tmp = dist

        # if tmp > 25:  # the therehold of tmp can be modifid according to your data   30
        if tmp > 50:  # the therehold of tmp can be modifid according to your data   30
            take_action = 1
        else:
            take_action = 0

        # modify the action
        action = np.load(f'{data_path}/actions/{idx}_actions.npy')
        action_pick = ((action[0] + action[2]) / 2)
        action_place = ((action[1] + action[3]) / 2)

        a = lambda x: math.floor((x - 58) / 10) if math.floor((x - 58) / 10) > 0 else 0
        if take_action:
            action = np.array([a(action_pick[0]), a(action_pick[1]), a(action_place[0]), a(action_place[1])])
        else:
            action = np.array([a(action_place[0]), a(action_place[1]), a(action_place[0]), a(action_place[1])])

        #         action=np.array([0,0,0,0])

        d = (rgbd_before, rgbd_after, take_action, action)
        all_list.append(d)
    print('the len is ', len(all_list))
    #     print(all_list[0])
    return all_list




# path=r'/home/heruhan/gnq/paper_codes/FabricFlowNet/biman_towel_actpickplace_n500_h3_co1_am0.9_tc0.3_cam0.65_lact'
data_path=r'/home/heruhan/gnq/paper_codes/DeFNet/biman_towel_actpickplace_n100_h3_co1_am0.9_tc0.3_cam0.65_lact_20221108_12_12_24'
out=creat_dataset_rgbd_resize_actionlabel(data_path)
store_path=r'/home/heruhan/gnq/paper_codes/DeFNet/lsr/datasets/'

try:
    os.mkdir(store_path)
    print('create the path')
except:
    print('the path is exist')

name='folding_task_2500.pkl'

with open(store_path+name,'wb') as f:
    pickle.dump(out,f)

print("done")