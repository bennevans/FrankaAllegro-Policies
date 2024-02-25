# Main script for hand interractions 
import numpy as np
from .env import DexterityEnv

class StateRecognitionR1B0R(DexterityEnv):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([0.41281763, -0.18762423,  0.31808898, -0.244461  , -0.6497351 ,
       -0.6484832 ,  0.31233397]), 
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])
        )