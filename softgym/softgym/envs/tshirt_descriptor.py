import numpy as np
from gym import spaces
import pyflex
from softgym.envs.cloth_env import FlexEnv
from softgym.action_space.action_space import PickerPickPlace
from softgym.utils.gemo_utils import *
from copy import deepcopy
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
import random

# def get_rotation_matrix(angle, axis):
# 	axis = axis / np.linalg.norm(axis)
# 	s = np.sin(angle)
# 	c = np.cos(angle)
#
# 	m = np.zeros((4, 4))
#
# 	m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
# 	# m[0][1] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
# 	m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
# 	# m[0][2] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
# 	m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
# 	m[0][3] = 0.0
#
# 	# m[1][0] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
# 	m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
# 	m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
# 	# m[1][2] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
# 	m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
# 	m[1][3] = 0.0
#
# 	# m[2][0] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
# 	m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
# 	# m[2][1] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
# 	m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
# 	m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
# 	m[2][3] = 0.0
#
# 	m[3][0] = 0.0
# 	m[3][1] = 0.0
# 	m[3][2] = 0.0
# 	m[3][3] = 1.0
#
# 	return m

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def uv_to_world_pos(camera_params, depth, u, v, particle_radius=0.0075, on_table=False):
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees

    # from cam coord to world coord
    cam_x, cam_y, cam_z = camera_params['default_camera']['pos'][0], camera_params['default_camera']['pos'][1], camera_params['default_camera']['pos'][2]
    cam_x_angle, cam_y_angle, cam_z_angle = camera_params['default_camera']['angle'][0], camera_params['default_camera']['angle'][1], camera_params['default_camera']['angle'][2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0]) 
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z
    matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    z = depth[int(np.rint(u)), int(np.rint(v))]
    if on_table or z == 0:
        vec = ((v - x0) / fx, (u - y0) / fy)
        z = (particle_radius - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])
    else:
        # adjust for particle radius from depth image
        z -= particle_radius
        
    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    
    cam_coord = np.ones(4)
    cam_coord[:3] = (x, y, z)
    world_coord = matrix @ cam_coord

    return world_coord

def particle_uv_pos(camera_params, m ): #particle_uv_pos(camera_params,None)
    """
    camera_params:
    {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 720, 'height': 720}}
    2209=47*47
    """
    # uv=[]
    # h=camera_params['default_camera']['height']
    # w=camera_params['default_camera']['width']
    # for i in range(47):
    #     for j in range(47):
    #         tmp=[160+(400/46)*j,160+(400/46)*i]
    #         uv.append(tmp)
    # uv = np.array(uv)
    # return uv
    # print(pyflex.get_positions().reshape((-1, 4)[:, [0,2]]))
    particle_pos = pyflex.get_positions().reshape((-1, 4))[:, 0:3]
    particle_pos = particle_pos [:,[0,2,1]]
    particle_pos = particle_pos [:, 0:2]
    # print(particle_pos.shape)
    # print(particle_pos)
    k=0.14375
    bias=k/200*360
    particle_pos=particle_pos+bias
    uv = (particle_pos/k) * 200
    # print(uv)
    return uv


def random_sample_from_masked_image(mask, k):#random_sample_from_masked_image(mask, 1)
    grasp_points=[]
    x,y=mask.shape

    for i in range(x):
        for j in range(y):
            if mask[i][j]:
                grasp_points.append([i,j])
    if len(grasp_points)==0:
        return [[0,0],[0,0]]
    # if k==2:
    #     return [[random.randint(0,x-1),random.randint(0,x-1)],[random.randint(0,y-1),random.randint(0,y-1)]]
    # if k==1:
    #     return [[random.randint(0,x-1),0],[random.randint(0,y-1),0]]
    if k==1:
        grasp_1=random.randint(0,(len(grasp_points)-1))
        grasp_1=grasp_points[grasp_1]
        return [[grasp_1[0],0],[grasp_1[1],0]]
    elif k==2:
        grasp_1 = random.randint(0, (len(grasp_points) - 1))
        grasp_2 = random.randint(0, (len(grasp_points) - 1))
        grasp_1 = grasp_points[grasp_1]
        grasp_2 = grasp_points[grasp_2]
        return [[grasp_1[0], grasp_2[0]], [grasp_1[1], grasp_2[1]]]
