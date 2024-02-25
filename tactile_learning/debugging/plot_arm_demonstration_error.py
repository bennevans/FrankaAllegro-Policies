import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from tqdm import tqdm 

from transform_utils import *
from tactile_learning.utils import turn_images_to_video

np.set_printoptions(precision=2, suppress=True)

# Script that reads the saved deployment data and plots and turns all of them to a video
def load_deployment(deployment_info_file):
    deployment_info = np.load(deployment_info_file)
    print('deployment_info.shape: {}'.format(deployment_info.shape))
    actions, states = deployment_info[0], deployment_info[1]

    return actions[:,-7:], states[:,-7:]

def get_angle_difference(action, state):

    if np.dot(action, state) < 0.0:
        state = -state
    debug_quat_diff = quat_distance(action, state)
    angle_diff = 180*np.linalg.norm(quat2axisangle(debug_quat_diff))/np.pi

    return angle_diff

def get_mult_angle_difference(actions, states):
    angle_diffs = []
    for i in range(len(actions)):
        angle_diffs.append(
            get_angle_difference(actions[i,:], states[i,:])
        )

    return np.asanyarray(angle_diffs)

def get_error_stats(deployment_dump_dir):
    deployment_info_file = os.path.join(deployment_dump_dir, 'openloop_traj.npy')
    actions, states = load_deployment(deployment_info_file)

    avg_error = np.zeros(4)
    max_error = np.ones(4) * -sys.maxsize
    min_error = np.ones(4) * sys.maxsize

    print('actions.shape: {}'.format(actions.shape))
    for i in range(0, len(actions)-1):
        for j in range(4):

            if j == 3:
                # Plot the Angle difference
                error = get_mult_angle_difference(
                    actions = actions[i:i+1, j:],
                    states = states[i+1:i+2, j:]
                )
                
            else:
                error = np.abs(actions[i,j] - states[i+1,j])

            # print('avg_error[{}]: {}, error: {}'.format(j, avg_error[j], error))
            avg_error[j] += error
            if error > max_error[j]:
                max_error[j] = error
            if error < min_error[j]:
                min_error[j] = error

    avg_error /= len(actions)-1

    print('AVG ERROR: {}, MAX ERROR: {}, MIN ERROR: {}'.format(
        avg_error, max_error, min_error
    ))


def plot_all_sets_of_data(deployment_dump_dir):
    deployment_info_file = os.path.join(deployment_dump_dir, 'openloop_traj.npy')
    actions, states = load_deployment(deployment_info_file)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    pbar = tqdm(total=len(actions))
    print('actions.shape: {}'.format(actions.shape))
    step_size = 15
    for i in range(0, len(actions), step_size):
        for j in range(4):
            
            if i > 50*step_size:
                indices_to_get = np.arange(i+1-(50*step_size), i+1, step_size)
            else:
                indices_to_get = np.arange(0, i+1, step_size)


            if j == 3:
                # Plot the Angle difference
                angle_diffs = get_mult_angle_difference(
                    actions = actions[indices_to_get, j:],
                    states = states[indices_to_get+1, j:]
                )
                axs[int(j/2), j%2].plot(angle_diffs)
                axs[int(j/2), j%2].set_title(f'Angle Difference')
            else:

                error = np.abs(actions[indices_to_get,j] - states[indices_to_get+1,j])

                axs[int(j/2), j%2].plot(error, label='error')


                axs[int(j/2), j%2].set_title(f'{j}th Axes')
                axs[int(j/2), j%2].legend()

        pbar.update(step_size)
        os.makedirs(f'{deployment_dump_dir}/arm_traj_replay_error', exist_ok=True)
        plt.savefig(os.path.join(f'{deployment_dump_dir}/arm_traj_replay_error/state_{str(i).zfill(5)}.png'))
        # plt.pause(0.1)

        plt.close()
        # plt.clf()
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))


def remove_all_files(viz_dir):
    files = glob.glob(f'{viz_dir}/*.png')
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    # deployment_dump_dir = '/home/irmak/Workspace/tactile-learning/deployment_data/infra_test/yinlong/R0B1B/26_trans_250_rot_250_rate_60hz_trans_scale_15_rot_scale_1_preprocessed_a0.01_h0.01'
    deployment_dump_dir = '/home/irmak/Workspace/tactile-learning/deployment_data/infra_test/yinlong/R0B1B/20_trans_200_rot_250_rate_60hz_trans_scale_10_rot_scale_2'
    get_error_stats(deployment_dump_dir)
    plot_all_sets_of_data(
        deployment_dump_dir=deployment_dump_dir
    )

    turn_images_to_video(
        viz_dir = os.path.join(f'{deployment_dump_dir}/arm_traj_replay_error'), 
        video_fps = 20,
    )

    remove_all_files(viz_dir=os.path.join(f'{deployment_dump_dir}/arm_traj_replay_error'))

