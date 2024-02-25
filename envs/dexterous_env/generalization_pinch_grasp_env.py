# Main script for hand interractions 
import numpy as np
from .env import DexterityEnv

class GeneralizationPinchGrasp(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([0.575466, -0.17820767, 0.23671454, -0.281564, 
                               -0.6797597,  -0.6224841,  0.2667619]), 
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])
        )