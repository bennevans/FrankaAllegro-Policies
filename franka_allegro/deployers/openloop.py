# Script to deploy already created demo
import numpy as np
import os 

from openteach.constants import *

from .deployer import Deployer
from franka_allegro.utils import load_data

class OpenLoop(Deployer):
    def __init__(
        self,
        data_path, # root in string
        data_representations,
        demo_to_run,
        apply_hand_states = False, # boolean to indicate if we should apply commanded allegro states or actual allegro states
        deployment_dump_dir = None
    ):

        super().__init__(data_path=data_path, data_representations=data_representations)
        self._set_data(demos_to_use=[demo_to_run])

        print('self.data.keys(): {}'.format(self.data.keys()))

        self.state_id = 0
        self.hand_action_key = 'hand_joint_states' if apply_hand_states else 'hand_actions'

        self.deployment_dump_dir = deployment_dump_dir
        if not deployment_dump_dir is None:
            self.deployment_info = dict(
                actions = [],
                states = []
            )

    def get_action(self, **kwargs):

        action = dict()

        if 'allegro' in self.data_reprs:
            demo_id, action_id = self.data[self.hand_action_key]['indices'][self.state_id] 
            hand_action = self.data[self.hand_action_key]['values'][demo_id][action_id] # Get the next commanded action (commanded actions are saved in that timestamp)

        if 'franka' in self.data_reprs or 'kinova' in self.data_reprs:
            demo_id, arm_id = self.data['arm']['indices'][self.state_id] 
            arm_action = self.data['arm']['values'][demo_id][arm_id] # Get the next saved kinova_state

        for data in self.data_reprs:
            if data == 'allegro':
                action[data] = hand_action
            if data == 'franka' or data == 'kinova':
                action[data] = arm_action

        self.state_id += 1

        # If it is required to save the deployment then we will add it to the deployment_info
        if not self.deployment_dump_dir is None:
            state_dict = kwargs['state_dict']
            self.deployment_info['actions'].append(np.concatenate([
                action['allegro'], action['franka'] 
            ], axis=0))
            self.deployment_info['states'].append(np.concatenate([
                state_dict['allegro'], state_dict['franka']
            ], axis=0))

            print('per frame shapes: actions - {}, states - {}'.format(
                np.concatenate([
                    action['allegro'], action['franka'] 
                ], axis=0).shape,
                np.concatenate([
                    state_dict['allegro'], state_dict['franka']
                ], axis=0).shape
            ))

        return action

    def save_deployment(self): # We don't really need to do anything here
        # Turn the deployment_info to a numpy array
        if not self.deployment_dump_dir is None:

            os.makedirs(self.deployment_dump_dir, exist_ok=True)

            self.deployment_info['actions'] = np.stack(self.deployment_info['actions'], axis=0)
            self.deployment_info['states'] = np.stack(self.deployment_info['states'], axis=0)

            deployment_info = np.stack(
                [self.deployment_info['actions'], self.deployment_info['states']],
                axis=0
            )

            print('deployment_info.shape: {}'.format(deployment_info.shape))
            np.save(os.path.join(
                self.deployment_dump_dir, 'openloop_traj.npy'
            ), deployment_info)