import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

class XelaCurvedPlotter():
    def __init__(self, display_plot=True):
        if not display_plot:
            matplotlib.use('Agg')

        thumb = [['thumb_empty'],
          ['thumb_tip'],
          ['thumb_section2'],
          ['thumb_section3']]

        index = [['index_tip'],
                ['index_section1'],
                ['index_section2'],
                ['index_section3']]

        ring = [['ring_tip'],
                ['ring_section1'],
                ['ring_section2'],
                ['ring_section3']]

        middle = [['mid_tip'],
                ['mid_section1'],
                ['mid_section2'],
                ['mid_section3']]

        all_fingers = [thumb, index, middle, ring]

        hand = [[thumb, index, middle, ring],
                ['palm', 'palm', 'palm', 'palm']]
        
        fig, self.axs = plt.subplot_mosaic(hand, figsize=(10,20))

    def _set_limits(self):
        pass

    def plot_tactile_sensor(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
    # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (240, 240, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []
        for j in range(60, 180+1, 40): # Y
            for i in range(60, 180+1, 40): # X - It goes from top left to bottom right row first 
                tactile_coordinates.append([i,j])

        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20), # NOTE: Change this
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis

    def plot_tactile_curved_tip(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
        # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (240, 240, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []
        for j in range(20, 240, 40): # y axis
            # x axis is somewhat hard coded
            for i in range(20, 240, 40):
                if j == 20 and (i == 100 or i == 140): # Only the middle two will be added
                    tactile_coordinates.append([i,j])
                elif (j > 20 and j < 100) and (i > 20 and i < 220):
                    tactile_coordinates.append([i,j])
                elif j >= 100: 
                    tactile_coordinates.append([i,j])
        
        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20),
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis


    def plot_tactile_palm(self,ax, sensor_values, use_img=False, img=None, title='Tip Position'):
        # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
        img_shape = (480, 960, 3) # For one sensor
        blank_image = np.ones(img_shape, np.uint8) * 255
        if use_img == False: 
            img = ax.imshow(blank_image.copy())
        ax.set_title(title)

        # Set the coordinates for each circle
        tactile_coordinates = []

        for j in range(70, 190+1, 40):
            for i in range(220, 420+1, 40):
                tactile_coordinates.append([i,j])

        for j in range(70, 190+1, 40):
            for i in range(540, 740+1, 40):
                tactile_coordinates.append([i,j])

        for j in range(270, 390+1, 40):
            for i in range(540, 740+1, 40):
                tactile_coordinates.append([i,j])

        # Plot the circles 
        for i in range(sensor_values.shape[0]):
            center_coordinates = (
                tactile_coordinates[i][0] + int(sensor_values[i,0]/20),
                tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
            )
            radius = max(10 + int(sensor_values[i,2]/10), 2)
        
            if i == 0:
                frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
            else:
                frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

        img.set_array(frame_axis)

        return img, frame_axis

    def draw(self, palm_sensor_values, fingertip_sensor_values, finger_sensor_values, figure_plot_path):
        cnt_fingertip=0
        cnt_finger=0
        for k in self.axs:
            if 'tip' in k:
                self.fingertip_sensor_values=fingertip_sensor_values
                self.plot_tactile_curved_tip(self.axs[k], sensor_values=self.fingertip_sensor_values[cnt_fingertip], title=k)
                cnt_fingertip+=1
            elif 'palm' in k:
                palm_sensor_values = np.concatenate(palm_sensor_values, axis=0)
                assert palm_sensor_values.shape == (72,3), f'palm_sensor_values.shape: {palm_sensor_values.shape}'
                self.plot_tactile_palm(self.axs[k], sensor_values = palm_sensor_values, title=k)
            elif not 'empty' in k:
                self.finger_sensor_values= finger_sensor_values
                self.plot_tactile_sensor(self.axs[k], sensor_values=self.finger_sensor_values[cnt_finger], title=k)
                cnt_finger+=1
            self.axs[k].get_yaxis().set_ticks([])
            self.axs[k].get_xaxis().set_ticks([])

        plt.savefig(figure_plot_path, bbox_inches='tight')

        # Resetting and pausing the plot
        plt.pause(0.01)
        plt.cla()