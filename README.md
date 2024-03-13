# FrankaAllegro-Policies

This repository contains the policy learning code for the Franka-Allegro Robot environment with [TAVI](https://arxiv.org/abs/2309.12300). 

This codebase has been released as a part of [OpenTeach](https://arxiv.org/abs/2403.07870)


Clone the repository using the following command.

`git clone https://github.com/NYU-robot-learning/FrankaAllegro-Policies.git`

Installation

`mamba env create -f conda_env.yml`

Install the Codebase as a module using 
`pip install -e .`

1. Collect Demos using [OpenTeach]().
2. Change the path in [configs](/franka_allegro/configs/preprocess.yaml) to the path where you saved your data.
3. Preprocess the data using the following command
`python3 preprocess.py`
4. Choose the path you have collected data in [configs](/franka_allegro/configs/train.yaml) in `data_dir:`. Choose the data representations you want to use for training from `image/tactile,allegro,franka`
5. Once preprocessed you can train the Vision and tactile encoder using 
`python train.py`. You can edit `train.yaml` accordingly with the choice of encoder, rl_learners, rewarders and optimizers.
6. After training the Vision and tactile encoders you can start the offset learning following [TAVI](https://arxiv.org/abs/2309.12300) using 
`python train_online.py`.
7. You can set the task, base_policy,agent, explorer and rewarder. [configs](/franka_allegro/configs/train_online.yaml)

### Citation
If you use this repo in your research, please consider citing the paper as follows:
```@misc{iyer2024open,
      title={OPEN TEACH: A Versatile Teleoperation System for Robotic Manipulation}, 
      author={Aadhithya Iyer and Zhuoran Peng and Yinlong Dai and Irmak Guzey and Siddhant Haldar and Soumith Chintala and Lerrel Pinto},
      year={2024},
      eprint={2403.07870},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}