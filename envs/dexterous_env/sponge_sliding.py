# Main script for hand interractions 
import numpy as np
from .env import DexterityEnv

class SpongeSliding(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([0.5849516 , -0.21625192,  0.31164503, -0.29108885, -0.64113265,
       -0.6541724 ,  0.276178]), 
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])
        )