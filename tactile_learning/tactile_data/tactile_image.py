import numpy as np
import random
import torch

import torch.nn.functional as F
import torchvision.transforms as T

from copy import deepcopy as copy

from franka_allegro.utils import tactile_clamp_transform, tactile_scale_transform

# Class to retrieve tactile images depending on the type
class TactileImage:
    def __init__(
        self,
        tactile_image_size=224,
        shuffle_type=None
    ):
        self.shuffle_type = shuffle_type
        self.size = tactile_image_size

        self.transform = T.Compose([
            T.Resize(tactile_image_size),
            T.Lambda(tactile_clamp_transform),
            T.Lambda(tactile_scale_transform)
        ])

    def get(self, type, tactile_values):
        if type == 'whole_hand':
            return self.get_whole_hand_tactile_image(tactile_values)
        if type == 'single_sensor':
            return self.get_single_tactile_image(tactile_values)
        if type == 'stacked':
            return self.get_stacked_tactile_image(tactile_values)

    def get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15,4,4,3) # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0,3,1,2))
        tactile_image = tactile_image.reshape(-1,4,4)
        return self.transform(tactile_image)

    def get_single_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value) # tactile_value.shape: (16,3)
        tactile_image = tactile_image.view(4,4,3)
        tactile_image = torch.permute(tactile_image, (2,0,1))
        return self.transform(tactile_image) 

    def get_whole_hand_tactile_image(self, tactile_values):
        # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = F.pad(tactile_image, (0,0,0,0,1,0), 'constant', 0)
        # reshape it to 4x4
        tactile_image = tactile_image.view(16,4,4,3)

        pad_idx = list(range(16))
        if self.shuffle_type == 'pad':
            random.seed(10)
            random.shuffle(pad_idx)
            
        tactile_image = torch.concat([
            torch.concat([tactile_image[pad_idx[i*4+j]] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)

        if self.shuffle_type == 'whole':
            copy_tactile_image = copy(tactile_image)
            sensor_idx = list(range(16*16))
            random.seed(10)
            random.shuffle(sensor_idx)
            for i in range(16):
                for j in range(16):
                    rand_id = sensor_idx[i*16+j]
                    rand_i = int(rand_id / 16)
                    rand_j = int(rand_id % 16)
                    tactile_image[i,j,:] = copy_tactile_image[rand_i, rand_j, :]

        tactile_image = torch.permute(tactile_image, (2,0,1))

        return self.transform(tactile_image)
    
    def get_tactile_image_for_visualization(self, tactile_values):
        tactile_image = self.get_whole_hand_tactile_image(tactile_values)
        tactile_image = T.Resize(224)(tactile_image) # Don't need another normalization
        tactile_image = (tactile_image - tactile_image.min()) / (tactile_image.max() - tactile_image.min())
        return tactile_image  



class TactileImageCurved(TactileImage):
    def __init__(
        self,
        tactile_image_size, # With the curved readings for the compatbility to be the same we are making it a square
        shuffle_type=None
    ):
        tactile_image_size = (tactile_image_size, tactile_image_size) # We want to make sure that it gets resized to squares
        super().__init__(tactile_image_size=tactile_image_size, shuffle_type=shuffle_type)

    def get(self, type, tactile_values):
        # tactile_values: dict(finger_values=(...), fingertip_values=(..), palm_values=(..))
        # print('IN TACTILE IMAGE CURVED - TACTILE VALUES KEYS: {}'.format(
        #     tactile_values.keys()
        # ))
        return self.get_whole_hand_tactile_image(
            finger_values = tactile_values['finger_values'],
            fingertip_values = tactile_values['fingertip_values'],
            palm_values = tactile_values['palm_values']
        ) # NOTE: For now we don't have any other shuffle types - we might just not need them?

    def _get_fingertip_stacked_image(self, fingertip_values):
        curr_fingertip_values = torch.FloatTensor(fingertip_values)
        top_row = torch.concat([
            F.pad(curr_fingertip_values[finger_id, :2], (0,0,2,2)) 
            for finger_id in range(4)], dim=0).unsqueeze(0)

        sec_row = torch.concat([
            F.pad(curr_fingertip_values[finger_id, 2:6], (0,0,1,1))
            for finger_id in range(4)], dim=0).unsqueeze(0)

        tip_rest = torch.stack([
                torch.concat([
                    curr_fingertip_values[finger_id, (6+6*row_id):(6+6*(row_id+1))]
                    for finger_id in range(4)], dim=0)
            for row_id in range(4)], dim=0)

        fingertip_stacked = torch.concat([
            top_row, sec_row, tip_rest], dim=0)

        return fingertip_stacked
    
    def _get_fingertip_images(self, fingertip_values): # Instead of merging all of them will be giving them separately
        # fingertip_values: (4, 30, 3)
        curr_fingertip_values = torch.FloatTensor(fingertip_values)

        # Concatenate the top row
        top_row = torch.stack([
            F.pad(curr_fingertip_values[finger_id, :2], (0,0,2,2)) 
            for finger_id in range(4)], dim=0).unsqueeze(1)

        sec_row = torch.stack([
            F.pad(curr_fingertip_values[finger_id, 2:6], (0,0,1,1))
            for finger_id in range(4)], dim=0).unsqueeze(1)

        tip_rest = torch.stack([
                torch.stack([
                    curr_fingertip_values[finger_id, (6+6*row_id):(6+6*(row_id+1))]
                    for finger_id in range(4)], dim=0)
            for row_id in range(4)], dim=0)

        fingertip_images = torch.concat(
            [top_row, sec_row, tip_rest], dim=1)

        # Returns (6, 24, 3) images
        return fingertip_images


    def _get_all_finger_images(self, finger_values, fingertip_values):
        # fingertip_values: (4, 30, 3), finger_values: (11, 24, 3)
        curr_finger_values = torch.FloatTensor(finger_values).view(11,4,4,3)
        # Pad each of the finger section readings to have similar readings as the tips
        curr_finger_values = F.pad(curr_finger_values, (0,0,1,1,1,1))

        # Add fingertips to these to have a final (15,6,6,3) reading
        fingertip_images = self._get_fingertip_images(fingertip_values)
        thumb_fingers = torch.concat([fingertip_images[0:1], curr_finger_values[:2]], dim=0)
        other_fingers = torch.concat([
            torch.concat([fingertip_images[i+1:i+2], curr_finger_values[2+(3*i):2+(3*(i+1))]], dim=0)
            for i in range(3)], dim=0)
        all_finger_images = torch.concat([thumb_fingers, other_fingers], dim=0)

        return all_finger_images
    
    
    def get_whole_hand_tactile_image(self, palm_values, finger_values, fingertip_values):
        # palm_values: (3, 24, 3), fingertip_values: (4, 30, 3), finger_values: (11, 24, 3)
        
        # Get the (15,6,6,3) tactile images for all the finger segments including the tip
        all_finger_images = self._get_all_finger_images(
            finger_values = finger_values, 
            fingertip_values = fingertip_values)

        # Pad this accordingly as we used
        padded_finger_images = F.pad(all_finger_images, (0,0,0,0,0,0,1,0))

        # Merge all the fingers together
        all_fingers_image = torch.concat([
            torch.concat([padded_finger_images[i*4+j] for j in range(4)], dim=0)
            for i in range(4)
        ], dim=1)


        # Get the palm image - we pad them to fill out the emptiness of 
        curr_palm_values = torch.FloatTensor(palm_values).view(3,4,6,3)
        top_rows = torch.concat( # Top two palm pads 
            [F.pad(curr_palm_values[i,:], (0,0, 3,3, 1,1)) for i in range(2)],
            dim=1) 
        bot_rows = F.pad(curr_palm_values[-1,:], (0,0, 15,3, 1,1)) # Bottom palm pad
        palm_image = torch.concat([top_rows, bot_rows], dim=0)

        # Merge fingers images and palm together
        whole_hand_image = torch.concat([all_fingers_image, palm_image], dim=0)
        whole_hand_image = torch.permute(whole_hand_image, (2,0,1))
        return self.transform(whole_hand_image)
    
    def get_tactile_image_for_visualization(self, tactile_values):
        tactile_image = self.get_whole_hand_tactile_image(
            palm_values=tactile_values['palm_values'],
            finger_values=tactile_values['finger_values'],
            fingertip_values=tactile_values['fingertip_values'])
        tactile_image = T.Resize(224)(tactile_image) # Don't need another normalization
        tactile_image = (tactile_image - tactile_image.min()) / (tactile_image.max() - tactile_image.min())
        return tactile_image
        # npimg = tactile_image.numpy()
        # return np.transpose(npimg, (1, 2, 0)) # When this is passed to plt.imshow() it should work
