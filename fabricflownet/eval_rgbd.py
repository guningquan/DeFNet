import os
import time 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pyflex
from softgym.envs.bimanual_env import BimanualEnv
from softgym.envs.bimanual_tshirt import BimanualTshirtEnv

from fabricflownet.flownet.models import FlowNet
from fabricflownet.picknet.models import FlowPickSplitModel
from fabricflownet.picknet.dataset import Goals

class EnvRollout(object):
    def __init__(self, args):
        self.args = args

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if 'towel' in args.cloth_type:
            self.env = BimanualEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    use_desc=False,
                    action_repeat=1,
                    headless=args.headless,
                    shape='default' if 'square' in args.cloth_type else 'rect')
        elif args.cloth_type == 'tshirt':
            self.env = BimanualTshirtEnv(use_depth=True,
                    use_cached_states=False,
                    use_desc=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless)

        goal_data = Goals(cloth_type=args.cloth_type)
        self.goal_loader = DataLoader(goal_data, batch_size=1, shuffle=False, num_workers=0)
        self.save_dir = f'{args.run_path}/rollout'
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def load_model(self, load_iter=0):
        picknet_cfg = OmegaConf.load(f'{self.args.run_path}/config.yaml')
        first_path = f"{self.args.run_path}/weights/first_{load_iter}.pt"
        second_path = f"{self.args.run_path}/weights/second_{load_iter}.pt"

        self.picknet = FlowPickSplitModel(
            s_pick_thres=self.args.single_pick_thresh,
            a_len_thres=self.args.action_len_thresh).cuda()
        self.picknet.first.load_state_dict(torch.load(first_path))
        self.picknet.second.load_state_dict(torch.load(second_path))
        self.picknet.eval()

        # flow model
        self.flownet = FlowNet(input_channels=8).cuda()
        checkpt = torch.load(f'{os.path.dirname(__file__)}/data/{picknet_cfg.flow}')
        self.flownet.load_state_dict(checkpt['state_dict'])
        self.flownet.eval()

    def run(self, crumple_idx=None):
        """Main eval loop
        crumple_idx: which crumpled configuration to load whenever environment is reset. -1 is no crumpling
        """
        actions = []
        total_metrics = []
        times = []

        for goal_idx, b in enumerate(self.goal_loader):
            for step in range(len(b)):
                transition = b[step]
                goal_name = transition['goal_name'][0]
                coords_pre = transition['coords_pre'].squeeze()
                coords_post = transition['coords_post'].squeeze()
                goal_im = transition['goal_im'].cuda()

                goal_im=goal_im.squeeze(0).permute(2,0,1)

                if step == 0:
                    self.env.reset()
                    pyflex.set_positions(coords_pre)
                    pyflex.step()
                
                for rep in range(self.args.goal_repeat):
                    self.env.render(mode='rgb_array')  #/home/heruhan/gnq/paper_codes/FabricFlowNet/softgym/softgym/envs/flex_env.py
                    rgb, depth = self.env.get_rgbd(cloth_only=True)   #/home/heruhan/gnq/paper_codes/FabricFlowNet/softgym/softgym/envs/bimanual_env.py
                    depth = cv2.resize(depth, (200, 200))
                    # print('the size of depth is',depth.shape)
                    # print('the max value of depth is',depth.max())
                    # plt.imshow(depth, cmap='gray')
                    # plt.show()
                    rgb = cv2.resize(rgb, (200, 200))/255
                    # plt.imshow(rgb, cmap='gray')
                    # plt.show()
                    # print('the size of rgb is',rgb.shape)
                    # print('the max value of rgb is',rgb.max())
                    #############
                    depth = np.concatenate((rgb, depth[:, :, None]), axis=-1)
                    #############
                    # curr_im = torch.FloatTensor(depth).unsqueeze(0).cuda()
                    curr_im = torch.FloatTensor(depth).permute(2,0,1).cuda()
                    # print('the curr_im size is',curr_im.shape)
                    #############
                    # print('the goal_im size is', goal_im.shape)
                    # goal_im = goal_im.permute(2,0,1)
                    # print('the goal_im size is', goal_im.shape)
                    # goal_im= torch.stack([goal_im,goal_im,goal_im,goal_im],1).squeeze(0)
                    #############

                    inp = torch.cat([curr_im, goal_im]).unsqueeze(0)
                    # print('the inp size is', inp.shape)
                    flow_out = self.flownet(inp)
                    # print('the flow_out size is', flow_out.shape)
                    # mask flow
                    flow_out[0,0,:,:][inp[0,0,:,:] == 0] = 0
                    flow_out[0,1,:,:][inp[0,0,:,:] == 0] = 0

                    start = time.time()
                    # print('the size of curr_im.unsqueeze(0) is',curr_im.unsqueeze(0).shape)
                    # print('the size of goal_im.unsqueeze(0) is',goal_im.unsqueeze(0).shape)
                    action, unmasked_pred = self.picknet.get_action(flow_out, curr_im.unsqueeze(0), goal_im.unsqueeze(0))
                    duration = time.time() - start
                    times.append(duration)

                    actions.append(action)
                    next_state, reward, done, _ = self.env.step(action, pickplace=True, on_table=True)
                    self.env.render(mode='rgb_array')
                    
                    #########################trick###################################
                    depth_img=next_state['depth']

                    # path_save=f'{self.save_dir}/{goal_name}-step{step}-rep{rep}.png'
                    # cv2.imwrite(path_save, depth_img)
                    # print(depth_img.max())
                    # print(depth_img.min())
                    # plt.imshow(depth_img, cmap='gray')
                    # plt.show()
                    
                    ############################################################

                    img = cv2.cvtColor(next_state["color"], cv2.COLOR_RGB2BGR)
                    img = self.action_viz(img, action, unmasked_pred)           # write the flow into img
                    cv2.imwrite(f'{self.save_dir}/{goal_name}-step{step}-rep{rep}.png', img)

            metrics = self.env.compute_reward(goal_pos=coords_post[:,:3])
            total_metrics.append(metrics)
            print(f"goal {goal_idx}: {metrics}")

        print(f"average action time: {np.mean(times)}")
        print('the total_metrics is',total_metrics)
        print("\nmean, std metrics: ",np.mean(total_metrics), np.std(total_metrics))
        np.save(f"{self.save_dir}/actions.npy",actions)
        return total_metrics

    def action_viz(self, img, action, unmasked_pred):
        ''' img: cv2 image
            action: pick1, place1, pick2, place2
            unmasked_pred: pick1_pred, pick2_pred'''
        pick1, place1, pick2, place2 = action
        pick1_pred, pick2_pred = unmasked_pred

        # draw the masked action
        u1,v1 = pick1
        u2,v2 = place1
        cv2.circle(img, (int(v1),int(u1)), 6, (0,200,0), 2)
        cv2.arrowedLine(img, (int(v1),int(u1)), (int(v2),int(u2)), (0,200,0), 3)
        u1,v1 = pick2
        u2,v2 = place2
        cv2.circle(img, (int(v1),int(u1)), 6, (0,200,0), 2)
        cv2.arrowedLine(img, (int(v1),int(u1)), (int(v2),int(u2)), (0,200,0), 3)
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', help='Run to evaluate', required=True)
    parser.add_argument('--ckpt', help="checkpoint to evaluate, don't set to evaluate all checkpoints", type=int, default=-1)
    parser.add_argument('--cloth_type', help='cloth type to load', default='square_towel', choices=['square_towel', 'rect_towel', 'tshirt'])
    parser.add_argument('--single_pick_thresh', help='min px distance to switch dual pick to single pick', default=300)
    parser.add_argument('--action_len_thresh', help='min px distance for an action', default=10)
    parser.add_argument('--goal_repeat', help='Number of times to repeat one goal', default=1)
    parser.add_argument('--seed', help='random seed', default=0)
    parser.add_argument('--headless', help='Run headless evaluation', action='store_true')
    parser.add_argument('--crumple_idx', help='index for crumpled initial configuration, set to -1 for no crumpling', type=int, default=-1)
    args = parser.parse_args()

    run_name = args.run_path.split('/')[-1]
    output_dir = '/'.join(args.run_path.split('/')[:-1])

    avg_metrics = []
    full_mean = []

    env = EnvRollout(args)

    # loop through the checkpoints to evaluate
    rng = range(0, 300001, 5000) if args.ckpt==-1 else range(args.ckpt, args.ckpt+1, 5000)
    for i in rng:
        print(f"loading {i}")
        try:
            env.load_model(load_iter=i)
            fold_mean = env.run()
            full_mean.append(fold_mean)
            print(f"mean: {np.mean(fold_mean)}")
            avg_metrics.append([i, np.mean(fold_mean), np.std(fold_mean)])
        except EOFError:
            print("EOFError. skipping...")

    avg_metrics = np.array(avg_metrics)
    idx = avg_metrics[:,1].argmin()
    print(f"\nmin: {avg_metrics[idx,0]} fold mean/std: {avg_metrics[idx,1]*1000.0:.3f} {avg_metrics[idx, 2]*1000.0:.3f}")

    
    if args.cloth_type == 'square_towel':
        ms_idx = 40
    elif args.cloth_type == 'rect_towel':
        ms_idx = 3
    elif args.cloth_type == 'tshirt':
        ms_idx = 2

    full_mean = np.array(full_mean)*1000.0
    print(f"\nall: {np.mean(full_mean):.3f} {np.std(full_mean):.3f}")
    # print(f"one-step: {np.mean(full_mean[idx,:ms_idx]):.3f} {np.std(full_mean[idx,:ms_idx]):.3f}")
    # print(f"mul-step: {np.mean(full_mean[idx,ms_idx:]):.3f} {np.std(full_mean[idx,ms_idx:]):.3f}")

    with open(f'{env.save_dir}/metrics.txt', "w") as f:
        f.write(f'all mean/std: {np.mean(full_mean):.3f} {np.std(full_mean):.3f}')
        # f.write(f'one-step mean/std: {np.mean(full_mean[idx,:ms_idx]):.3f} {np.std(full_mean[idx,:ms_idx]):.3f}')
        # f.write(f'mul-step mean/std: {np.mean(full_mean[idx,ms_idx:]):.3f} {np.std(full_mean[idx,ms_idx:]):.3f}')

    np.save(f'{env.save_dir}/fold_metrics.npy', avg_metrics)
    np.save(f'{env.save_dir}/all_goals.npy', full_mean)