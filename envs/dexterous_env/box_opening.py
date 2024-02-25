# Main script for hand interractions 
import numpy as np
from .env import DexterityEnv

class MusicBoxOpening(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([ 0.61939764, -0.21312736,  0.21798871, -0.30505708, -0.6352805 ,
       -0.640904  ,  0.30430415]), 
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])
        )