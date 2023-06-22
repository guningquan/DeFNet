# DeFNet
[Arxiv](https://arxiv.org/abs/2303.00323)
## Installation
1. Clone this repo.
2. We persuade Ubuntu18.04 system and python3.6.
3. Create the environment and activate it: <code> conda env create -f environment.yml</code>
4. Setup the SoftGym according to the blog https://danieltakeshi.github.io/2021/02/20/softgym/
    - You should have installed the Docker.
    - Pull the pre-built docker file. `sudo docker pull xingyu/softgym`
    - We have to start a container, and you can modify the path according to your computer.
        ```
        sudo nvidia-docker runI am running a few minutes late; my previous meeting is running over.\
        -v /home/heruhan/gnq/paper_codes/DeFNet/softgym:/workspace/softgym \
        -v /home/heruhan/anaconda3:/home/heruhan/anaconda3 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -it xingyu/softgym:latest bash
        ```
    - Now you are in the Docker environment. Go to the softgym directory and compile PyFlex

        ```
        export PATH="/home/heruhan/anaconda3/bin:$PATH"
        cd softgym
        source activate DeFNet
        . ./prepare_1.0.sh && ./compile_1.0.sh
        ```
    - When you open a new Terminal every time, I persuade you to input the following commands:
        ```
        export PYTORCH_JIT=0
        export PYFLEXROOT=${PWD}/softgym/PyFlex
        export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PWD}/DPI-Net:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
        export LD_LIBRARY_PATH=${PYFLEXROOT}/   external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
        ```

## Trainning
### Data Generation
You can generate your dataset through the SoftGym, the repo provide the script to help you get the fabric dataset easily. Besides, you can modify the script according to your tasks.
<p><code>python dataset_generation/collect_the_data.py</code></p>

### Preprocess Datasets
Preprocess the dataset into a suitable format to train the MM module.
```
python lsr/dataset_generation.py
python lsr/lsrv2_preprocess_datasets.py
```

### Training the MM
```
python lsr/train_MM.py --exp_vae="folding_task_MM"
```
### Training the FlowNet
Modify the config.yaml in flownet document.
```
python fabricflownet/flwonet/train.py
```


### Training the PickNet
Modify the config.yaml in picknet document.
```
python fabricflownet/picknet/train.py
```

## Prediction
### Reason the intermediate states of the folding task
In this process, we can get the intermediate states of cloth by inputting the initial and the goal.
```
python lsr/lsrv2_folding_planning.py --example="folding" --save_path='examples' --dataset_path=''  --model_path='models'
```
### Calculate the actions of manipulation
We provide two prediction methods, RGBD or depth information.
```
# reason the manipulation actions through depth images.
python fabricflownet_depth/eval.py --run_path=fabricflownet_depth/data_depth/picknet_run --ckpt=105000 --cloth_type=square_towel

# reason the manipulation actions through RGBD images.
python fabricflownet/eval_rgbd.py --run_path=fabricflownet/data/picknet_run --ckpt=105000 --cloth_type=square_towel
```
