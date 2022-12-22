from __future__ import print_function
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import cv2
import pickle
import random
from os import path
from lsrv2_examples_utils import get_example_vap, perform_lsrv2_building, perform_mapping
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example' , type=str, default='folding', help='Type of task: rigid_normal, rigid_hard or hybrid')
    parser.add_argument('--from_scratch' , type=bool, default=False, help='produce from scratch')
    parser.add_argument('--save_path',type=str, default='examples', help='The path to save examples')
    parser.add_argument('--dataset_path',type=str, default='xxx', help='The path of dataset')
    parser.add_argument('--model_path',type=str, default='models', help='The path of MM model')
    parser.add_argument('--set_goal',type=str, default='', help='set your goal')
    args = parser.parse_args()

    example_type=args.example
    from_scratch=args.from_scratch
    save_path='lsr/'+ args.save_path
    dataset_path=args.dataset_path
    model_path='lsr/'+ args.model_path
    set_goal=args.set_goal

    if (not os.path.isdir(save_path)):
        os.makedirs(save_path)

    if example_type=='folding':
        print("******producing folding example******")
        # models:
        train_dataset_name="folding_task_2500"
        holdout_dataset_name="foldingd_task_holdout"
        mapping_module="folding_task_MM"
        apm_seed = 1977
        
        # set hyperparams:
        distance_type=1
        # c_max=20   #5
        c_max=80   #5
        rng=random.randint(0,1337)

    mm_output_file="./"+save_path+"/mm_" + mapping_module + "_data_" + train_dataset_name + ".pkl"
    if from_scratch or not path.exists(mm_output_file):

        train_dataset_name_1=dataset_path+'/'+train_dataset_name

        latent_map= perform_mapping(train_dataset_name_1,mapping_module,model_path)
        # save
        with open(mm_output_file, 'wb') as f:
            pickle.dump(latent_map, f)
        print("--- mapping done ---")
    else:
        print(" found: " + mm_output_file + " already exist, using it.")
        file = open(mm_output_file,'rb')
        latent_map = pickle.load(file)
        file.close()

    # build lsrv2
    lsr_output_name="./"+save_path+"/lsrv2_" +  mapping_module + "_data_" + train_dataset_name + ".pkl"
    if from_scratch or not path.exists(lsr_output_name):
        lsrv2=perform_lsrv2_building(latent_map,distance_type,c_max)
        # save
        with open(lsr_output_name, 'wb') as f:
            pickle.dump(lsrv2, f)
        print("--- lsrv2 building done ---")
        

    else:
        print(" found: " + lsr_output_name + " already exists, using it.")
        # load
        file = open(lsr_output_name,'rb')
        lsrv2 = pickle.load(file)
        file.close()




    # example_visual_action_plan=get_example_vap(example_type,lsrv2,mapping_module,action_prediction_module,
    #                                            apm_seed,holdout_dataset_name,rng)
    example_visual_action_plan=get_example_vap(example_type,lsrv2,mapping_module,None,
                                               apm_seed,holdout_dataset_name,rng,model_path,set_goal)

    now = datetime.now()
    timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    example_type_rename=example_type+'_'+ timestr
    cv2.imwrite("./results/"+example_type_rename + ".png",example_visual_action_plan)
    print("**** produced " + example_type_rename + ".png")


if __name__== "__main__":
  main()