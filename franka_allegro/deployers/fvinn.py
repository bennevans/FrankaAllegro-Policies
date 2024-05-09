from franka_allegro.deployers.vinn import VINN
import torch
from franka_allegro.utils import load_dataset_image
import numpy as np

class FVINN(VINN):
    def __init__(self, algo_config, augmentation_functions, index_to_factor_map, task_factors, device, *args, **kwargs):
        algo, cfg = algo_config
        self.algo = algo.to(device)
        self.color_aug, self.geom_aug = augmentation_functions
        self.index_to_factor_map = np.array(index_to_factor_map)
        self.task_factors = task_factors
        task_indices = [np.where(self.index_to_factor_map == factor)[0][0] for factor in task_factors]
        self.task_factor_mask = torch.zeros(len(index_to_factor_map), dtype=torch.bool, device=device)
        self.task_factor_mask[task_indices] = True
        self.device = device
        super().__init__(*args, **kwargs)

    def _get_all_representations(self):
        # TODO: implement saving representations
        super()._get_all_representations()
    
    def _get_one_representation(self, repr_data):
        image = repr_data['image'].to(self.device).unsqueeze(0)
        factor_mask = self.task_factor_mask.unsqueeze(0)
        with torch.no_grad():
            enc_info = self.algo.encoder(image, factor_mask)
        factors = enc_info['factors'].squeeze()
        relevant_factors = factors[factor_mask.squeeze()]
        relevant_flat = relevant_factors.flatten().cpu().numpy()
        return relevant_flat
    
    def _set_encoders(self, image_out_dir=None, image_model_type=None, tactile_out_dir=None, tactile_model_type=None):
        super()._set_encoders(image_out_dir, image_model_type, tactile_out_dir, tactile_model_type)

    def _load_dataset_image(self, demo_id, image_id):
        dset_img = load_dataset_image(self.data_path, demo_id, image_id, self.view_num)
        img_np = np.asarray(dset_img)
        img_geom = self.geom_aug(image=img_np)['image']
        img_color = self.color_aug(image=img_geom)['image']
        return img_color