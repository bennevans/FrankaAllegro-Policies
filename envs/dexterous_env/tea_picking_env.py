import numpy as np 
from .env import DexterityEnv 

class TeaPicking(DexterityEnv):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def set_home_state(self):
        self.home_state = dict(
            franka = np.array([ 0.53916454, -0.23157556,  0.25026545, -0.1642998 , -0.48592967,
                -0.8089102 ,  0.28730178]),
            allegro = np.array([
                0, -0.17453293, 0.78539816, 0.78539816,           # Index
                0, -0.17453293,  0.78539816,  0.78539816,         # Middle
                0.08726646, -0.08726646, 0.87266463,  0.78539816, # Ring 
                1.04719755,  0.43633231,  0.26179939, 0.78539816  # Thumb
            ])                    
        )