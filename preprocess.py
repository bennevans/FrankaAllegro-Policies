import glob
import hydra 
from omegaconf import DictConfig

# from franka_allegro.datasets import *

@hydra.main(version_base=None, config_path='franka_allegro/configs', config_name='preprocess')
def main(cfg : DictConfig) -> None:

    # Initialize the preprocessor module
    prep_module = hydra.utils.instantiate(cfg.preprocessor_module)
    prep_module.apply()

if __name__ == '__main__':
    main()