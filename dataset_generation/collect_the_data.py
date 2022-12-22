from softgym.envs.bimanual_env import BimanualEnv
from softgym.envs.bimanual_tshirt import BimanualTshirtEnv
import numpy as np
import pyflex
import softgym.envs.tshirt_descriptor as td
import torch
import os
from PIL import Image
import json
import random
import cv2
from collections import namedtuple
import copy
import matplotlib.pyplot as plt
import socket
from datetime import datetime
#from edge_masker import EdgeMasker
from utils import get_harris
from itertools import *
# modify the defination of depth 20221031
import argparse

Experience = namedtuple('Experience', ('obs', 'goal', 'act', 'rew', 'nobs', 'done'))

class DatasetGenerator(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        
        if cfgs['cloth_type'] == 'towel':
            self.env = BimanualEnv(use_depth=cfgs['img_type'] == 'depth',
                    use_cached_states=False,
                    horizon=cfgs['horizon'],
                    use_desc=False,
                    action_repeat=1,
                    headless=cfgs['headless'],
                    rect=cfgs['rect'])

        elif cfgs['cloth_type'] == 'tshirt':
            self.env = BimanualTshirtEnv(use_depth=cfgs['img_type'] == 'depth',
                    use_cached_states=False,
                    use_desc=False,
                    horizon=cfgs['horizon'],
                    action_repeat=1,
                    headless=cfgs['headless'])

        print("env created")

        # self.em = EdgeMasker(self.env, cfgs['cloth_type'], tshirtmap_path=None, edgethresh=cfgs['edgethresh'])

        # particle_num = pyflex.get_n_particles()
        # print("the particles number is",particle_num)

        self.corners = self.get_corner_particles()

    def makedirs(self):
        save_folder = os.path.join(self.cfgs['dataset_folder'], self.cfgs['dataset_name'])
        if self.cfgs['debug']:
            os.system('rm -r %s' % save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            os.makedirs(os.path.join(save_folder, 'images'))
            os.makedirs(os.path.join(save_folder, 'actions'))
            os.makedirs(os.path.join(save_folder, 'coords'))
            # os.makedirs(os.path.join(save_folder, 'descs'))
            os.makedirs(os.path.join(save_folder, 'image_masks'))
            os.makedirs(os.path.join(save_folder, 'rendered_images'))
            os.makedirs(os.path.join(save_folder, 'knots'))
            # os.makedirs(os.path.join(save_folder, 'edge_masks'))
        return save_folder

    def get_masked(self, img):
        """Just used for masking goals, otherwise we use depth"""
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.array([0., 15., 0.]), np.array([255, 255., 255.]))
        kernel = np.ones((3,3),np.uint8)
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return morph
    #
    # def get_rgbd(self, cloth_only=False):
    #     if cloth_only:
    #         rgb, depth = pyflex.render_cloth()
    #     else:
    #         rgb, depth = pyflex.render()
    #     rgb = np.array(rgb).reshape(self.env.camera_height, self.env.camera_width, 4)
    #     rgb = rgb[::-1, :, :]  # reverse the height dimension
    #     rgb = rgb[:, :, :3]
    #     depth = np.array(depth).reshape(self.env.camera_height, self.env.camera_width)
    #     depth = depth[::-1, :]  # reverse the height dimension
    #
    #     print("the size of depth is",depth.shape)
    #     print('the min depth is', depth.min())
    #     print('the max depth is', depth.max())
    #     print('the value is',depth[10][10])
    #     print('the value is', depth[360][360])
    #     plt.imshow(depth, cmap='gray')
    #     plt.show()
    #     depth[depth >= 999] = 0  # use 0 instead of 999 for null
    #
    #     print("the size of depth is",depth.shape)
    #     print('the min depth is',depth.min())
    #     print('the max depth is',depth.max())
    #     plt.imshow(depth, cmap='gray')
    #     plt.show()
    #
    #     mask = depth > 0
    #     print("the size of mask is",mask.shape)
    #     print('the min mask is', mask.min())
    #     print('the max mask is', mask.max())
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()
    #     return rgb, depth, mask
        # rgbd = pyflex.render_sensor()
        # rgbd, _ = pyflex.render()
        # rgbd = np.array(rgbd).reshape(self.env.camera_height, self.env.camera_width, 4)
        # rgbd = rgbd[::-1, :, :]
        # rgb = rgbd[:, :, :3]
        # img = self.env.get_image(self.env.camera_height, self.env.camera_width)
        # depth = rgbd[:, :, 3]
        # print("#"*100)
        # print('the depth is',depth)
        # print('the shape of depth is', depth.shape)
        # print('the max depth is',depth.max())
        # print('the min depth is',depth.min())
        # print("#" * 100)
        # mask = depth > 0
        # return img, depth, mask
    #
    def get_rgbd(self):
        _, depth = pyflex.render()

        depth=depth.reshape(self.env.camera_height, self.env.camera_width)

        # depth=np.rot90(depth.T, 1)  #  or we can use depth[::-1,:] to reverse
        depth=depth[::-1, :]

        img = self.env.get_image(self.env.camera_height, self.env.camera_width)
        # img=np.rot90(img,2)
        #
        # print("#"*100)
        # print('the depth is',depth)
        # print('the shape of depth is', depth.shape)
        # print('the max depth is',depth.max())
        # print('the min depth is',depth.min())
        # print("#" * 100)

        depth_desk = depth.max()

        depth[depth >= depth_desk] = 0

        # for i in range(self.env.camera_height):
        #     for j in range(self.env.camera_width):
        #         depth[i][j] = depth_desk - depth[i][j]

        # print("#"*100)
        # print('the depth is',depth)
        # print('the shape of depth is', depth.shape)
        # print('the max depth is',depth.max())
        # print('the min depth is',depth.min())
        # print("#" * 100)

        mask = depth > 0
        return img, depth, mask


    def line_pt_dir(self, a,b,p):
        ax,ay = a
        bx,by = b
        px,py = p

        bx -= ax
        by -= ay
        px -= ax
        py -= ay

        cross_prod = bx * py - by * px

        # right of line
        if cross_prod > 0:
            return 1

        # left of line
        if cross_prod < 0:
            return -1

        # on the line
        return 0


    def get_rand_action(self, state, img, depth, edgemasks, coords, action_type='perp', single_scale=True, debug_idx=None):
        clothmask = depth > 0
        ##############corners####################
        cloth_corners_labels = np.zeros((200, 200), dtype=float)
        for k in range(4):
            if k == 0:
                for i in range(4):
                    for j in range(4):
                        cloth_corners_labels[44 + i][44 + j] = 1
            if k == 1:
                for i in range(4):
                    for j in range(4):
                        cloth_corners_labels[151 + i][44 + j] = 1
            if k == 2:
                for i in range(4):
                    for j in range(4):
                        cloth_corners_labels[44 + i][151 + j] = 1
            if k == 3:
                for i in range(4):
                    for j in range(4):
                        cloth_corners_labels[151 + i][151 + j] = 1
        mask=cloth_corners_labels*clothmask
        # for i in range(200):
        #     for j in range(200):
        #         if (cloth_corners_labels[i][j]) and (clothmask[i][j]):
        #             mask[i][j] = 1
        #         else:
        #             mask[i][j] = 0
        ########################################
        # if np.random.uniform() < self.cfgs['actmaskprob']:
        #     if self.cfgs['use_corner']:
        #         # print('the clothmask is',clothmask.astype(np.float32))
        #         harris_corners = get_harris(clothmask.astype(np.float32))
        #         true_corners = self.get_true_corner_mask(clothmask)
        #         # print("*"*100)
        #         # print('the number harris_corners is',np.sum(harris_corners))
        #         # print('the number true_corners is',np.sum(true_corners))
        #         # print("*"*100)
        #         if np.sum(true_corners) > 2 and np.random.uniform() < self.cfgs['truecratio']:
        #             mask = true_corners > 0
        #         elif np.sum(harris_corners) > 2:
        #             mask = harris_corners > 0
        #         else:
        #             mask = clothmask
        #     else:
        #         all_mask, fge_mask, ce_mask = edgemasks
        #
        #         if np.sum(ce_mask != 0) > 2 and np.random.uniform() < self.cfgs['cemaskratio']: # sample from cloth edge mask
        #             mask = ce_mask > 0
        #         else: # sample from fg edge mask
        #             mask = fge_mask > 0
        # else: # Cloth mask
        #     mask = clothmask
        
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(depth)
        # ax[1].imshow(mask)
        # plt.show()
        

        if action_type == 'qnet':
            # sample until valid action found
            while True:
                pick_idx = td.random_sample_from_masked_image(mask, 1)
                u,v = pick_idx[0][0], pick_idx[1][0]
                angle_idx = np.random.randint(0,8)
                angle = np.deg2rad(angle_idx * 45)
                width_idx = np.random.randint(3)
                width = width_idx * 25.0

                pick_u1 = int(np.clip(u + np.sin(angle) * width, 10, 190))
                pick_v1 = int(np.clip(v + np.cos(angle) * width, 10, 190))
                pick_u2 = int(np.clip(u - np.sin(angle) * width, 10, 190))
                pick_v2 = int(np.clip(v - np.cos(angle) * width, 10, 190))

                if mask[pick_u1, pick_v1] and mask[pick_u2, pick_v2]:
                    break

            print(f"qnet act: {u},{v} angle: {angle} width: {width}")
            print(f"u1,v1 {pick_u1, pick_v1} u2,v2 {pick_u2, pick_v2}")

            # fold toward center
            if self.line_pt_dir([pick_u1, pick_v1],[pick_u2, pick_v2],[100,100]) < 0:
                fold_dir = angle - (np.pi/2)
            else:
                fold_dir = angle + (np.pi/2)

            # sample fold length
            dist = np.random.uniform(25,75)
            place_u1 = int(np.clip(pick_u1 + dist * np.sin(fold_dir), 10, 190))
            place_v1 = int(np.clip(pick_v1 + dist * np.cos(fold_dir), 10, 190))
            place_u2 = int(np.clip(pick_u2 + dist * np.sin(fold_dir), 10, 190))
            place_v2 = int(np.clip(pick_v2 + dist * np.cos(fold_dir), 10, 190))

            pick1 = [pick_u1, pick_v1]
            place1 = [place_u1, place_v1]
            pick2 = [pick_u2, pick_v2]
            place2 = [place_u2, place_v2]

            return np.array([angle_idx,width_idx,u,v]), np.array([pick1, place1, pick2, place2])

        if action_type == 'pickplace':
            # returns two arrays of x, and y positions with num_pick number of values
            # print('the masks shape is',mask.shape)
            ##########################################################################
            # print("the size of mask is",mask.shape)
            # print('the mask is ',mask)
            # num_corner=0
            # num_mask_no=0
            # for i in range(mask.shape[0]):
            #     for j in range(mask.shape[1]):
            #         if mask[i][j]:
            #             num_corner+=1
            #         else:
            #             num_mask_no+=1
            # print('the number of corner is', num_corner)
            # print('the number of corner_negative is', num_mask_no)
            ##########################################################################
            ''' 
            #the real grasp two points
            pick_idx = td.random_sample_from_masked_image(mask, 2)
            pick_u1,pick_v1 = pick_idx[0][0], pick_idx[1][0]
            pick_u2,pick_v2 = pick_idx[0][1], pick_idx[1][1]
            
            angle = np.arctan2(pick_u1 - pick_u2, pick_v1 - pick_v2)
            # fold toward center
            if self.line_pt_dir([pick_u1, pick_v1],[pick_u2, pick_v2],[100,100]) < 0:
               angle -= (np.pi/2)
            else:
               angle += (np.pi/2)
            # if np.random.uniform() < 0.5:
            #     angle -= (np.pi/2) 
            # else:
            #     angle += (np.pi/2)

            # angle noise
            #angle += np.random.uniform(-np.pi/4,np.pi/4)

            dist = np.random.uniform(25,150) # default: 25,100

            # dist noise seperate for two pickers
            #dist1 = dist + np.random.uniform(0,30)
            #dist2 = dist + np.random.uniform(0,30)
            dist1 = dist2 = dist

            place_u1 = int(np.clip(pick_u1 + dist1 * np.sin(angle), 20, 180))
            place_v1 = int(np.clip(pick_v1 + dist1 * np.cos(angle), 20, 180))
            place_u2 = int(np.clip(pick_u2 + dist2 * np.sin(angle), 20, 180))
            place_v2 = int(np.clip(pick_v2 + dist2 * np.cos(angle), 20, 180))

            pick1 = [pick_u1, pick_v1]
            place1 = [place_u1, place_v1]
            pick2 = [pick_u2, pick_v2]
            place2 = [place_u2, place_v2]

            print(f"angle: {np.rad2deg(angle)} dist: {dist1} {dist2}")

            return np.array([pick1, place1, pick2, place2]), np.array([pick1, place1, pick2, place2])
            
            '''
            # grasp one points
            pick_idx = td.random_sample_from_masked_image(mask, 1)
            # print('the pick_idx is',pick_idx)
            pick_u1, pick_v1 = pick_idx[0][0], pick_idx[1][0]
            # print("pick_u1, pick_v1",pick_u1, pick_v1)
            x, y = mask.shape
            #########################################################################################
            # while True:
            #     angle_point_u,angle_point_v= random.randint(0,x-1),random.randint(0,y-1)
            #     # print(angle_point_u)
            #     # print(angle_point_v)
            #     # print(mask[angle_point_u,angle_point_v])
            #     # print(mask[angle_point_u+20, angle_point_v+20])
            #     # print(mask[angle_point_u-20, angle_point_v+20])
            #     # print(mask[angle_point_u-20, angle_point_v-20])
            #     # print(mask[angle_point_u+20, angle_point_v+20])
            #     if clothmask[np.clip(angle_point_u+5,0,199), np.clip(angle_point_v+5,0,199)] and clothmask[np.clip(angle_point_u-5,0,199), np.clip(angle_point_v+5,0,199)] :
            #         if clothmask[np.clip(angle_point_u-5,0,199), np.clip(angle_point_v-5,0,199)] and clothmask[np.clip(angle_point_u+5,0,199), np.clip(angle_point_v+5,0,199)] :
            #             if clothmask[angle_point_u,angle_point_v]:
            #                 break
            #########################################################################################

            dist = np.random.uniform(5, 100)
            place_map = np.zeros((200, 200), dtype=float)
            for i in range(200):
                for j in range(200):
                    if abs(dist - np.sqrt((pick_u1 - i) ** 2 + (pick_v1 - j) ** 2)) < 2:
                        place_map[i][j] = 1

            # print('the pick point is',pick_u1,pick_v1)
            # plt.imshow(place_map, cmap='gray')
            # plt.show()
            # plt.imshow(clothmask, cmap='gray')
            # plt.show()

            clothmask1=clothmask*place_map

            # plt.imshow(clothmask1, cmap='gray')
            # plt.show()

            place_points=[]
            for m in range(x):
                for n in range(y):
                    if clothmask1[m][n]:
                        place_points.append([m, n])

            if len(place_points)==0:
                pick1 = [pick_u1, pick_v1]
                pick2 = [pick_u1, pick_v1]
                return np.array([pick1, pick1, pick2, pick2]), np.array([pick1, pick1, pick2, pick2])

            angle_point = random.randint(0, (len(place_points) - 1))
            angle_point_u = place_points[angle_point][0]
            angle_point_v = place_points[angle_point][1]

            pick1 = [pick_u1, pick_v1]
            pick2 = [pick_u1, pick_v1]
            place1 = [angle_point_u, angle_point_v]
            place2 = [angle_point_u, angle_point_v]

            # pick1 = [152, 47]
            # pick2 = [152, 47]
            # place1 = [47, 152]
            # place2 = [47, 152]
            # print('the place point is', angle_point_u, angle_point_v)

            return np.array([pick1, place1, pick2, place2]), np.array([pick1, place1, pick2, place2])

        if action_type == 'debug':
            actions = [[[25,25],[40,85],[25,25],[40,85]],
                       [[25,25],[85,85],[25,25],[85,85]],
                       [[25,25],[85,40],[25,25],[85,40]],
                       [[25,25],[140,140],[25,25],[140,140]],
                       [[25,25],[90,160],[25,25],[90,160]],
                       [[25,25],[160,90],[25,25],[160,90]],

                       [[25,175],[85,115],[25,175],[85,115]],
                       [[25,175],[40,115],[25,175],[40,115]],
                       [[25,175],[80,165],[25,175],[80,165]],
                       [[25,175],[140,60],[25,175],[140,60]],
                       [[25,175],[90,35],[25,175],[90,35]],
                       [[25,175],[165,115],[25,175],[165,115]],

                       [[175,175],[110,110],[175,175],[110,110]],
                       [[175,175],[110,160],[175,175],[110,160]],
                       [[175,175],[160,110],[175,175],[160,110]],
                       [[175,175],[60,60],[175,175],[60,60]],
                       [[175,175],[35,60],[175,175],[35,60]],
                       [[175,175],[60,35],[175,175],[60,35]],

                       [[175,25],[110,90],[175,25],[110,90]],
                       [[175,25],[110,40],[175,25],[110,40]],
                       [[175,25],[160,90],[175,25],[160,90]],
                       [[175,25],[60,140],[175,25],[60,140]],
                       [[175,25],[35,90],[175,25],[35,90]],
                       [[175,25],[60,165],[175,25],[60,165]],

                       [[25,25],[80,25],[25,175],[80,175]],
                       [[25,25],[140,25],[25,175],[140,175]],
                       [[25,25],[25,80],[175,25],[175,80]],
                       [[25,25],[25,140],[175,25],[175,140]],
                       [[175,25],[120,25],[175,175],[120,175]],
                       [[175,25],[60,25],[175,175],[60,175]],
                       [[25,175],[25,120],[175,175],[175,120]],
                       [[25,175],[25,60],[175,175],[175,60]]]

            return np.array(actions[debug_idx]), np.array(actions[debug_idx])



    def save_data(self, idx, state, coords, img, depth, dataset_path,actions=None, beforeact=False):
        save_time = 'before' if beforeact else 'after'
        #all_mask, fge_mask, ce_mask = edgemasks
        mask = depth > 0
        uv = td.particle_uv_pos(self.env.camera_params,None)
        ###################################
        # print(self.env.camera_params)
        # np.save(os.path.join(dataset_path,'camera_params.npy'), self.env.camera_params)
        ###################################
        uv[:,[1,0]]=uv[:,[0,1]]
        # print("*"*200)
        # print(img)
        # print(img.max())
        # print(img.min())
        rgb_img = Image.fromarray(img, 'RGB')
        rgb_img.save(os.path.join(dataset_path, 'images', '%06d_rgb_%s.png'% (idx, save_time)))
        # #######################################################################################
        # print('the size of image_masks is',mask.shape)
        # print('the image_masks is ', mask)
        # num_mask = 0
        # num_mask_no = 0
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i][j]:
        #             num_mask += 1
        #         else:
        #             num_mask_no += 1
        # print('the number of mask is', num_mask)
        # print('the number of num_mask_no is', num_mask_no)
        # #######################################################################################
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(dataset_path, 'image_masks', '%06d_mask_%s.png' % (idx, save_time)))
        
        #if self.cfgs['desc_path']:
        #    desc_img = Image.fromarray(state['desc'])
        #    desc_img.save(os.path.join(dataset_path, 'descs', '%06d_desc_%s.png' % (idx, save_time)))

        np.save(os.path.join(dataset_path, 'rendered_images', '%06d_depth_%s.npy' % (idx, save_time)),depth)
        np.save(os.path.join(dataset_path, 'coords', '%06d_coords_%s.npy' % (idx, save_time)),coords)
        np.save(os.path.join(dataset_path, 'knots', '%06d_knots_%s.npy' % (idx, save_time)),uv)
        if beforeact==False:
            np.save(os.path.join(dataset_path, 'actions', '%06d_actions.npy' % idx), actions)
        #np.save(os.path.join(dataset_path, 'edge_masks', '%06d_allmask_%s.npy' % (idx, save_time)), all_mask)
        #np.save(os.path.join(dataset_path, 'edge_masks', '%06d_fgemask_%s.npy' % (idx, save_time)), fge_mask)
        #np.save(os.path.join(dataset_path, 'edge_masks', '%06d_cemask_%s.npy' % (idx, save_time)), ce_mask)

    def get_obs(self):
        coords = pyflex.get_positions().reshape(-1, 4)
        img, depth, mask = self.get_rgbd()

        if self.cfgs['use_corner']:
            edgemasks = None
        else:
            # all_mask, fge_mask, ce_mask = self.em.get_act_mask(self.env, coords, img, depth, mask)
            edgemasks = (all_mask, fge_mask, ce_mask)
        img = cv2.resize(img, (200, 200))
        depth = cv2.resize(depth, (200, 200))
        return coords, img, depth, edgemasks
        #return coords, img, depth

    def generate(self):
        min_reward = 0
        max_reward = -10000

        # load goals
        goals = []
        for g in self.cfgs['goals']:
            if g is not None:
                if self.cfgs['img_type'] == 'color':
                    goal = cv2.imread(f"../goals/{g}.png")
                    goal = cv2.cvtColor(goal, cv2.COLOR_BGR2RGB)
                    goal_mask = self.get_masked(goal) != 0
                    goal[goal_mask == False, :] = 0
                elif self.cfgs['img_type'] == 'depth':
                    goal = cv2.imread(f"../goals/{g}_depth.png")
                elif self.cfgs['img_type'] == 'desc':
                    goal = cv2.imread(f"../goals/{g}_desc.png")

                goal_pos = np.load('../goals/particles/{}.npy'.format(g))[:,:3]
            else:
                goal = g
                goal_pos = None
            goals.append([goal, goal_pos])

        save_folder = self.makedirs()
        #buf = []

        print(self.env.camera_params)
        np.save(os.path.join(save_folder, 'camera_params.npy'), self.env.camera_params)

        # check if dataset exists to resume
        if os.path.exists(os.path.join(save_folder,f'{self.cfgs["dataset_name"]}_idx.buf')):
            idx_buf = torch.load(os.path.join(save_folder,f'{self.cfgs["dataset_name"]}_idx.buf'))
            idx = len(idx_buf) - 1
            ep = idx // self.cfgs['horizon']
        else:
            idx_buf = [] # buffer with only indexes, no images
            idx = 0
            ep = 0

        while ep < self.cfgs['num_eps']:
            print("ep ",ep)
            goal, goal_pos = random.choice(goals)
            state = self.env.reset(given_goal=goal, given_goal_pos=goal_pos)
            done = False

            actions = []
            while not done:

                coords, img, depth, edgemasks = self.get_obs()

                # check if out of screen
                mask = depth > 0
                if np.sum(mask) < 250:
                    self.env.reset(given_goal=goal, given_goal_pos=goal_pos)
                    coords, img, depth = self.get_obs()

                self.save_data(idx, state, coords, img, depth, save_folder,actions=None, beforeact=True)

                buf_act, action = self.get_rand_action(state, img, depth, edgemasks, coords, action_type=self.cfgs['action_type'], debug_idx=ep)
                next_state, reward, done, _ = self.env.step(action, pickplace=True, on_table=self.cfgs['on_table'])
                actions.append(action)
                coords_next, img_next, depth_next, _ = self.get_obs()

                # check if out of screen
                mask = depth_next > 0
                if np.sum(mask) < 250:
                    self.env.reset(given_goal=goal, given_goal_pos=goal_pos)
                    continue

                self.save_data(idx, next_state, coords_next, img_next, depth_next, save_folder,action, beforeact=False)

                if reward < min_reward:
                    min_reward = reward
                if reward > max_reward:
                    max_reward = reward

                im_type = self.cfgs['img_type']
                #buf.append(Experience(state[im_type], state["goal"], action, reward, next_state[im_type], done))
                idx_buf.append(Experience(idx, None, buf_act, reward, idx, done))
            
                state = copy.deepcopy(next_state)
                self.env.render(mode='rgb_array')
                idx += 1
            print('all actions is',actions)
            actions_all = list(permutations(actions, len(actions)))
            for j in range(len(actions_all)):
                actions_sub = actions_all[j]
                state = self.env.reset(None, None)
                for k, action in enumerate(actions_sub):
                    # print('the action is',action)
                    coords, img, depth, edgemasks = self.get_obs()
                    self.save_data(idx, state, coords, img, depth, save_folder, actions=None, beforeact=True)

                    next_state, reward, done, _ = self.env.step(action, pickplace=True,
                                                                on_table=self.cfgs['on_table'])

                    coords_next, img_next, depth_next, _ = self.get_obs()
                    self.save_data(idx, next_state, coords_next, img_next, depth_next, save_folder, action,
                                   beforeact=False)
                    idx += 1

            if (ep % 500) == 0:
                print("saving...")
                #torch.save(buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}.buf'))
                torch.save(idx_buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}_idx.buf'))

            ep += 1
        #torch.save(buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}.buf'))
        print("saving...")
        torch.save(idx_buf, os.path.join(save_folder,f'{self.cfgs["dataset_name"]}_idx.buf'))

        # create knots info
        # print("create knots info...")
        # knots = os.listdir(os.path.join(save_folder, 'knots'))
        # knots.sort()
        # kdict = {}
        #################################################################
        # for i, name in enumerate(knots):
        #     knot = np.load(os.path.join(save_folder,'knots',name))
        #     knot = np.reshape(knot,(knot.shape[0],1,knot.shape[1]))
        #     kdict[str(i)] = knot.tolist()
        # with open(os.path.join(save_folder,'images','knots_info.json'),'w') as f:
        #     json.dump(kdict,f)
        #
        # print(f"min reward: {min_reward}, max reward: {max_reward}")
        # np.save(os.path.join(save_folder, f'rewards.npy'), [min_reward, max_reward])

        #################################################################

    def collect_goals(self):

        # Multi step goals
        names = ['opp_corn_in',
                'all_corn_in',
                'two_side_horz',
                'two_side_vert',
                'double_tri',
                'double_rect']


        opp_corn_in = [ [[25,25],[80,80],[25,25],[80,80]],
                        [[175,175],[120,120],[175,175],[120,120]]]

        all_corn_in = [ [[25,25],[80,80],[25,25],[80,80]],
                        [[25,175],[80,120],[25,175],[80,120]],
                        [[175,25],[120,80],[175,25],[120,80]],
                        [[175,175],[120,120],[175,175],[120,120]]]

        # two_side_horz = [[[25,25],[80,25],[25,175],[80,175]],
        #                  [[175,25],[120,25],[175,175],[120,175]]]
        #
        # two_side_vert = [[[25,25],[25,80],[175,25],[175,80]],
        #                  [[25,175],[25,120],[175,175],[175,120]]]

        double_tri = [ [[25,25],[175,175],[25,25],[175,175]],
                        [[25,175],[175,25],[25,175],[175,25]]]

        # double_rect = [ [[25,25],[165,25],[25,175],[165,175]],
        #                 [[110,30],[110,170],[170,30],[170,170]]]

        opp_corn_in = [ [[46,46],[89,89],[46,46],[89,89]],
                        [[153, 153], [109, 109], [153, 153], [109, 109]]]

        all_corn_in = [ [[46,46],[89,89],[46,46],[89,89]],
                        [[46,153],[89,109],[46,153],[89,109]],
                        [[153,46],[109,89],[153,46],[109,89]],
                        [[153,153],[109,109],[153,153],[109,109]]]

        double_tri = [ [[46,46],[153,153],[46,46],[153,153]],
                        [[46,153],[153,46],[46,153],[153,46]]]

        one_corn_in = [  [[46,46],[89,89],[46,46],[89,89]]]

        triangle = [ [[46,46],[153,153],[46,46],[153,153]] ]

        names = ['opp_corn_in']
        goals = [opp_corn_in]


        # prefix = "vsf_"
        prefix = " ms_"

        #goals = [(np.array(goal) * 0.851) + 15 for goal in goals]
        try:
            os.makedirs('goals/particles/')
        except:
            print('the dir exist')

        for name,actions in zip(names,goals):
            print("name:",name)

            goal, goal_pos = None, None
            state = self.env.reset(given_goal=goal, given_goal_pos=goal_pos)

            for i,action in enumerate(actions):

                # if name == 'test_goal':
                #     state = self.env.reset(given_goal=goal, given_goal_pos=goal_pos)

                # coords, img, depth, edgemasks = self.get_obs()
                # action = self.get_rand_action(state, img, depth, action_type='debug',edgemasks=None,coords=None,debug_idx=0)
                print('the action is',action)
                next_state, reward, done, _ = self.env.step(action, pickplace=True, on_table=self.cfgs['on_table'])


                coords_next, img_next, depth_next, edgemasks_next = self.get_obs()
                mask = depth_next > 0

                depth_next = depth_next*255
                depth_next = depth_next.astype(np.uint8)
                nobs = np.dstack([depth_next, depth_next, depth_next])
                nobs = cv2.resize(nobs, (200, 200))

                pos = pyflex.get_positions().reshape(-1, 4)

                impath = f'/home/heruhan/gnq/paper_codes/FabricFlowNet/goals/{prefix}{name}_{i}_depth.png'
                pospath = f'/home/heruhan/gnq/paper_codes/FabricFlowNet/goals/particles/{prefix}{name}_{i}.npy'

                cv2.imwrite(impath, nobs)
                np.save(pospath, pos)

                img_next = cv2.resize(img_next, (200, 200))
                rgb_img = Image.fromarray(img_next, 'RGB')

                rgb_img.save(f'/home/heruhan/gnq/paper_codes/FabricFlowNet/goals/{prefix}{name}_{i}.png')

    def get_corner_particles(self):
        state = self.env.reset(given_goal=None, given_goal_pos=None)

        uv = td.particle_uv_pos(self.env.camera_params,None)
        uv[:,[1,0]]=uv[:,[0,1]]
        uv = (uv/719) * 199

        # corner 1
        dists = np.linalg.norm((uv - np.array([25,25])),axis=1)
        c1 = dists.argmin()

        # corner 2
        dists = np.linalg.norm((uv - np.array([25,175])),axis=1)
        c2 = dists.argmin()

        # corner 3
        dists = np.linalg.norm((uv - np.array([175,175])),axis=1)
        c3 = dists.argmin()

        # corner 4
        dists = np.linalg.norm((uv - np.array([175,25])),axis=1)
        c4 = dists.argmin()

        # u,v = uv[c1]
        # print(u,v)
        # action = [[u,v],[175,175],[u,v],[175,175]]
        # self.env.step(action, pickplace=True)

        return c1,c2,c3,c4

    def get_particle_uv(self, idx):
        uv = td.particle_uv_pos(self.env.camera_params,None)
        uv[:,[1,0]]=uv[:,[0,1]]
        uv = (uv/719) * 199
        u,v = uv[idx]
        return u,v

    def get_true_corner_mask(self, clothmask, r=4):
        true_corners = np.zeros((200,200))
        for c in self.corners:
            b,a = self.get_particle_uv(c)
            h,w = true_corners.shape
            y,x = np.ogrid[-a:h-a, -b:w-b]
            cmask = x*x + y*y <= r*r
            true_corners[cmask] = 1

        true_corners = true_corners * clothmask
        # print("*" * 100)
        # print('the true_corner is',true_corners)
        # print("*" * 100)
        # plt.subplot(121)
        # plt.imshow(true_corners, cmap='gray')
        # plt.show()
        # print("the true_corners number is", true_corners.sum())
        return true_corners

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--headless', type=bool, default=True)
    # parser.add_argument('--epochs', type=bool, default=True)
    # args = parser.parse_args()
    # headless = args.headless

    num_eps = 100# 2200 # 3000
    horizon = 3  #2
    cloth_type = 'towel' # tshirt # towel
    action_type = 'pickplace' #qnet # pickplace # debug
    img_type = 'depth' # color # depth, # desc
    edgethresh = 10 if cloth_type == 'tshirt' else 5
    actmaskprob = 0.9 # 0.9
    cemaskratio = 0.0 # ratio of how often to sample cloth edge mask
    on_table = False
    truecratio = 0.3   # ratio of how often to sample from true corner  0.5
    use_corner = True # use corner vs edge mask
    now = datetime.now()
    timestr = now.strftime("%Y%m%d_%H_%M_%S")
    cfgs = {
        'debug': False, # overwrite old folder if True
        'num_eps': num_eps,
        'img_type': img_type,
        'cloth_type': cloth_type,
        'rect': False,
        'action_type': action_type,
        'edgethresh': edgethresh,
        'actmaskprob': actmaskprob,
        'cemaskratio': cemaskratio,
        'tshirtmap_path': None,
        'on_table': on_table,
        'horizon': horizon,
        'state_dim': 200*200*3,
        'dataset_folder': '',
        'action_dim': 7,
        'dataset_name': f'biman_{cloth_type}_act{action_type}_n{num_eps}_h{horizon}_co{int(use_corner)}_am{actmaskprob}_tc{truecratio}_cam0.65_lact_{timestr}',
        #'dataset_name': f'biman_tsh_act{action_type}_n{num_eps}_h{horizon}_co{int(use_corner)}_am{actmaskprob}_tc{truecratio}_cam0.65',
        'desc_path': False,
        'goals': [],
        'use_corner': use_corner,
        'truecratio': truecratio,
        'headless':True}

    cfgs['goals'] = [None]

    dataset = DatasetGenerator(cfgs)
    dataset.generate()
    # dataset.collect_goals()