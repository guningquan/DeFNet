from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
import torch
import numpy as np
import cv2
import pickle
import networkx as nx
import random
from lsrv2 import LSRV2
from datetime import datetime

def get_closest_nodes(G, z_pos_c1, z_pos_c2, distance_type=2):
    c1_close_idx=-1
    c2_close_idx=-1
    min_distance_c1=np.Inf
    min_distance_c2=np.Inf

    # find the closest nodes
    for g in G.nodes:
        tz_pos=G.nodes[g]['pos']
        node_distance_c1=np.linalg.norm(z_pos_c1-tz_pos, ord=distance_type)
        node_distance_c2=np.linalg.norm(z_pos_c2-tz_pos, ord=distance_type)
        if node_distance_c1<min_distance_c1:
            min_distance_c1=node_distance_c1
            c1_close_idx=g

        if node_distance_c2<min_distance_c2:
            min_distance_c2=node_distance_c2
            c2_close_idx=g
    return c1_close_idx, c2_close_idx


def perform_mapping(train_dataset_name, mapping_module,model_path):
    vae_config_file = os.path.join('.', 'lsr/configs', mapping_module + '.py')
    vae_directory = os.path.join('.', model_path, mapping_module)

    vae_config = SourceFileLoader(mapping_module, vae_config_file).load_module().config
    vae_config['exp_name'] = mapping_module
    vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
    
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])

    file = open("lsr/datasets/" + train_dataset_name +".pkl",'rb')

    dataset = pickle.load(file)
    file.close()
    
    vae_algorithm.load_checkpoint(vae_directory + "/vae_lastCheckpoint.pth")
    vae_algorithm.model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_map = []

    for i in range(len(dataset)):
        x=dataset[i][0]
        x2=dataset[i][1]
        ac=dataset[i][2]
        a_lam=dataset[i][3]        

        # x = torch.tensor(x/255.).float().permute(2, 0, 1).unsqueeze_(0).to(device)
        # x2 = torch.tensor(x2/255).float().permute(2, 0, 1).unsqueeze_(0).to(device)
        x = torch.tensor(x).float().permute(2, 0, 1).unsqueeze_(0).to(device)
        x2 = torch.tensor(x2).float().permute(2, 0, 1).unsqueeze_(0).to(device)
        
        # Encodings
        _, _, z, _=vae_algorithm.model.forward(x)
        _, _, z2, _=vae_algorithm.model.forward(x2)
        
        z_np=z[0,:].cpu().detach().numpy()
        z_np2=z2[0,:].cpu().detach().numpy()
        action = ac
        a_lambda = a_lam
        latent_map.append((z_np, z_np2, action, a_lambda,-1,-1))

    return latent_map


def perform_lsrv2_building(latent_map, distance_type, c_max):
    lsrv2 = LSRV2(latent_map, distance_type,c_max,  min_edge_w=0, min_node_m = 0,
                  directed_graph=False, a_lambda_format=None, verbose=False,lower_b=0, upper_b=3)
    lsr_obj, _, _, _ = lsrv2.optimize_lsr()
    return lsr_obj


def descale_coords(x):
        """
        Descales the coordinates from [0, 1] interval back to the original
        image size.
        """
        # data_max=np.array([2,2,2,2,2])
        # data_max=np.array([3,3,3,3,3])
        data_min=np.array([0,0,0,0,0])
        #
        data_max=np.array([13,13,13,13,13])
        # data_min=np.array([0,0,0,0,0])


        rescaled = x * (data_max - data_min) + data_min
        rounded_coords = np.around(rescaled).astype(int)

        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(data_min, np.minimum(rounded_coords, data_max))
        assert((cropped_rounded_coords >= data_min).all())
        assert((cropped_rounded_coords <= data_max).all())

        return cropped_rounded_coords.astype(int)
#
# def inpaint_actions(dec_imgs_paths, all_actions, example_type):
#     p_color=(255,0,0)
#     r_color=(0,255,0)
#     for i in range(len(dec_imgs_paths)):
#         img_path=dec_imgs_paths[i]
#         action_path=all_actions[i]
#
#         for j in range(len(img_path)-1):
#             t_img=img_path[j].copy()
#             action=action_path[j]
#             print('the last action is',action)
#             action_round=descale_coords(action)
#             print('the last action_round is', action_round)
#             # action specifications
#             if example_type=="hybrid":
#                 scx=58
#                 scy=75
#                 dx=67
#                 dy=67
#                 paint_pos=[(scx,scy),(scx+dx,scy),(scx+dx*2,scy),(scx,scy+dy),(scx+dx,scy+dy),
#                            (scx+dx*2,scy+dy),(scx,scy+dy*2),(scx+dx,scy+dy*2),(scx+dx*2,scy+dy*2)]
#
#                 #check if rope or not
#                 if action_round[0]==2:
#                     t_img = cv2.line(t_img, (10,10), (246,10), (71,70,213), 5)
#                 else:
#                     t_img = cv2.circle(t_img, paint_pos[action_round[1]*3+action_round[2]], 12, p_color, 4)
#                     t_img = cv2.circle(t_img, paint_pos[action_round[3]*3+action_round[4]], 8, r_color, -1)
#                 img_path[j]=t_img
#             else:
#                 # off_x=55
#                 # off_y=80
#                 # len_box=60
#                 off_x=58
#                 off_y=58
#                 len_box=10
#                 px=off_x+action_round[0]*len_box
#                 py=off_y+action_round[1]*len_box
#                 cv2.circle(t_img, (px,py), 12, p_color, 4)
#                 rx=off_x+action_round[3]*len_box
#                 ry=off_y+action_round[4]*len_box
#                 cv2.circle(t_img, (rx,ry), 8, r_color, -1)
#                 img_path[j]=t_img
#
#             dec_imgs_paths[i]=img_path
#     return dec_imgs_paths


def get_example_vap(example_type, lsrv2, mapping_module, action_prediction_module,
                                   apm_seed, holdout_dataset_name, rng,model_path,set_goal):
    # load holdout dataset
    # file = open("datasets/rgbd1103/" + holdout_dataset_name +".pkl",'rb')
    file = open("lsr/datasets/" + holdout_dataset_name +".pkl",'rb')
    dataset = pickle.load(file)
    file.close()

    #select a random start and goal
    random.seed(rng)
    # r_start_img=dataset[random.randint(0,len(dataset)-1)][0]/255.
    # r_start_img=dataset[random.randint(0,len(dataset)-1)][0]
    r_start_img=dataset[0][0]
    # r_goal_img=dataset[random.randint(0,len(dataset)-1)][0]/255.
    if set_goal:
        print('your data of customization')
        r_start_img = np.load(set_goal+'start.npy')
        r_goal_img=np.load(set_goal+'goal.npy')
    else:
        print('the random data')
        r_start_img = dataset[0][0]
        while(True):
            r_goal_img=dataset[random.randint(0,len(dataset)-1)][1]
            if (r_goal_img==r_start_img).any():
                break
    # print('the size of input is',r_goal_img.shape)
    # print(r_goal_img)
    # r_goal_img=dataset[14][1]
    # r_goal_img=dataset[11][0]

    #load VAE
    vae_config_file = os.path.join('.', 'lsr/configs', mapping_module + '.py')
    # vae_directory = os.path.join('.', 'models', mapping_module)
    vae_directory = os.path.join('.', model_path, mapping_module)

    vae_config = SourceFileLoader(mapping_module, vae_config_file).load_module().config
    vae_config['vae_opt']['exp_dir'] = vae_directory 
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    vae_algorithm.load_checkpoint(vae_directory + "/vae_lastCheckpoint.pth")
    vae_algorithm.model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load APM
    if action_prediction_module is not None:
        ap_config_file = os.path.join('.', 'configs', action_prediction_module + '.py')
        ap_directory = os.path.join('.', 'models', action_prediction_module+"/apnet_lastCheckpoint.pth")        
        #ap_directory = os.path.join('.', 'models', action_prediction_module+"/apnet_exeModel.pt")
        ap_config = SourceFileLoader(action_prediction_module, ap_config_file).load_module().config 
        ap_config['model_opt']['exp_dir'] = ap_directory 
        ap_algorithm = getattr(alg, ap_config['algorithm_type'])(ap_config['model_opt'])
        ap_algorithm.load_checkpoint('models/'+action_prediction_module + f"_seed{apm_seed}" +"/apnet_lastCheckpoint.pth")
        #ap_algorithm.load_best_model_pkl('models/'+action_prediction_module + f"_seed{apm_seed}" +"/apnet_exeModel.pt")
        
        ap_algorithm.model.eval()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # encode r_start and r_goal
    x = torch.tensor(r_start_img).float().permute(2, 0, 1).unsqueeze_(0).to(device)
    x2 = torch.tensor(r_goal_img).float().permute(2, 0, 1).unsqueeze_(0).to(device)

    # Encodings
    _, _, z, _=vae_algorithm.model.forward(x)
    _, _, z2, _=vae_algorithm.model.forward(x2)
    z_start=z[0,:].cpu().detach().numpy()
    z_goal=z2[0,:].cpu().detach().numpy()

    # find path
    # get closest start and goal node from graph
    [c1_close_idx, c2_close_idx] = get_closest_nodes(lsrv2.G2, z_start, z_goal,lsrv2.distance_type)
    
    # use graph to find paths
    paths=nx.all_shortest_paths(lsrv2.G2, source=c1_close_idx, target=c2_close_idx)

    z_paths_from_lsrv2=[]
    dec_imgs_paths=[]
    for path in paths:
        z_ts=[]
        d_imgs=[]
        for n_id in path:
            # print('the path is',n_id)
            z=lsrv2.G2.nodes[n_id]['pos']
            # decode z
            z = torch.from_numpy(z).float().to(device).unsqueeze(0)
            z_ts.append(z)
            
            img_rec,_=vae_algorithm.model.decoder(z)
            img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
            img_rec_cv=(img_rec_cv*255).astype("uint8")
            d_imgs.append(img_rec_cv)
        z_paths_from_lsrv2.append(z_ts)
        dec_imgs_paths.append(d_imgs)


    # save the result
    now = datetime.now()
    timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    dir_name = 'results/'+'results_' + timestr
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    os.makedirs(dir_name)
    for i in range(len(dec_imgs_paths)):
        sub_result = os.path.join(dir_name, f'road_{i}')
        sub_result_rgbd = os.path.join(dir_name, f'road_{i}', 'rgbd')
        sub_result_rgb = os.path.join(dir_name, f'road_{i}', 'rgb')
        sub_result_depth = os.path.join(dir_name, f'road_{i}', 'depth')
        os.makedirs(sub_result)
        os.makedirs(sub_result_rgbd)
        os.makedirs(sub_result_rgb)
        os.makedirs(sub_result_depth)

        start_rgbd_dir = os.path.join(sub_result_rgbd, 'start.npy')
        start_rgb_dir = os.path.join(sub_result_rgb, f'start.png')
        start_depth_dir = os.path.join(sub_result_depth, f'start_depth.png')

        goal_rgbd_dir = os.path.join(sub_result_rgbd, 'goal.npy')
        goal_rgb_dir = os.path.join(sub_result_rgb, f'goal.png')
        goal_depth_dir = os.path.join(sub_result_depth, f'goal_depth.png')

        np.save(start_rgbd_dir, cv2.resize(r_start_img, (200, 200))*255)
        cv2.imwrite(start_rgb_dir, cv2.resize(np.split(r_start_img, [-1], -1)[0]*255, (200, 200)))
        cv2.imwrite(start_depth_dir,cv2.resize(np.split(r_start_img, [-1], -1)[1]*255, (200, 200)))

        np.save(goal_rgbd_dir, cv2.resize(r_goal_img, (200, 200))*255)
        cv2.imwrite(goal_rgb_dir, cv2.resize(np.split(r_goal_img, [-1], -1)[0]*255, (200, 200)))
        cv2.imwrite(goal_depth_dir, cv2.resize(np.split(r_goal_img, [-1], -1)[1]*255, (200, 200)))

        for j in range(len(dec_imgs_paths[i])):
            rgbd = dec_imgs_paths[i][j]
            rgb = np.split(rgbd, [-1], -1)[0]
            depth = np.split(rgbd, [-1], -1)[1]
            # save the image to dir
            rgbd_dir = os.path.join(sub_result_rgbd, f'folding_{j}.npy')
            rgb_dir = os.path.join(sub_result_rgb, f'folding_{j}.png')
            depth_dir = os.path.join(sub_result_depth, f'folding_{j}_depth.png')
            rgbd=cv2.resize(rgbd, (200, 200))
            rgb=cv2.resize(rgb, (200, 200))
            depth = cv2.resize(depth, (200, 200))
            np.save(rgbd_dir, rgbd)
            cv2.imwrite(rgb_dir, rgb)
            cv2.imwrite(depth_dir, depth)
            # np.save(depth_dir, depth)


    # get actions with APM using zs
    # if action_prediction_module is not None:
    #     all_actions=[]
    #     for i in range(len(z_paths_from_lsrv2)):
    #         z_p=z_paths_from_lsrv2[i]
    #         path_action=[]
    #         for j in range(len(z_p)-1):
    #             z1_t=z_p[j]
    #             z2_t=z_p[j+1]
    #             action_to=ap_algorithm.model.forward(z1_t,z2_t)
    #             action=action_to.cpu().detach().numpy()
    #             action = np.squeeze(action)
    #             path_action.append(action)
    #         all_actions.append(path_action)
    #
    #     # inpaint images with actions (depennding on task)
    #     # print('the action is',all_actions)
    #     dec_imgs_paths=inpaint_actions(dec_imgs_paths,all_actions,example_type)

    # stack vaps 
    rows=[]
    buffer_img=(np.ones((r_start_img.shape[0],20,r_start_img.shape[2]))*255).astype("uint8")
    # with open("testout.pkl", 'wb') as f:
    #     pickle.dump(dec_imgs_paths, f)

    for row in dec_imgs_paths:
        t_row=[]
        t_row.append((r_start_img*255).astype("uint8"))
        t_row.append(buffer_img)
        for t_img in row:
            t_row.append((t_img))
            t_row.append(buffer_img)
        t_row.append((r_goal_img*255).astype("uint8"))
        # t_row.append((r_goal_img).astype("uint8"))
        x=np.concatenate([t_row[y] for y in range(len(t_row))],axis=1)
        rows.append(x)
    return np.concatenate([rows[y] for y in range(len(rows))],axis=0) 




