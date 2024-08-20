import os
import shutil
from typing import Tuple

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.nn as nn
from classeg.dataloading.datapoint import Datapoint
from classeg.extensions.super_resolution.utils.utils import get_forward_diffuser_from_config
from classeg.inference.inferer import Inferer
from classeg.utils.utils import read_json
from classeg.utils.constants import RESULTS_ROOT
from classeg.extensions.super_resolution.model.unstable_diffusion import UnstableDiffusion
from classeg.extensions.super_resolution.preprocessing.bitifier import bitmask_to_label
import albumentations as A
import pandas as pd

class SuperResolutionInferer(Inferer):
    def __init__(self,
                 dataset_id: str,
                 fold: int,
                 name: str,
                 weights: str,
                 input_root: str,
                 late_model_instantiation=True,
                 output_name=None,
                 infer_timesteps: int=1000,
                 ddim: bool=False,
                 batch: int=256,
                 **kwargs):
        """
        Inferer for pipeline.
        :param dataset_id: The dataset id used for training
        :param fold: The fold to run inference with
        :param weights: The name of the weights to load.
        """
        super().__init__(dataset_id, fold, name, weights, input_root, late_model_instantiation=late_model_instantiation)
        self.timesteps = self.config["max_timestep"]
        self.infer_timesteps = int(infer_timesteps)
        self.forward_diffuser = get_forward_diffuser_from_config(self.config, ddim=(self.infer_timesteps < 1000), timesteps=self.timesteps)
        self.ddim = False if ddim in ["0", "F", "False", "f", "false"] else True
        self.batch = int(batch)
        self.output_name = output_name
        self.kwargs = kwargs
        self.entries = []
        
        # self.model_json = read_json(f"{self.lookup_root}/model.json")

    def get_augmentations(self):
        def mask_transform(image=None, mask=None, **kwargs):
            mask = mask/mask.max()
            # make seg from RGBA [0, 1] to binary one channel
            mask = mask.sum(dim=1, keepdim=True)
            mask[mask > 0] = 1
            print("Mask")
            print(mask.min(), mask.max(), mask.dtype, mask.shape, np.unique(mask))
            return mask
        
        def image_transform(image=None, mask=None, **kwargs):
            # remove alpha
            image = image[..., 0:3]
            image = image/image.max()
            print("Image")
            print(image.min(), image.max(), image.dtype, image.shape)
            return image

        return None
        return A.Compose([
            A.ToFloat(),
            A.Lambda(mask=mask_transform, image=image_transform, p=1)
        ])

    def _get_model(self):
        """
        Loads the model and weights.
        :return:
        """
        if self.model is not None:
            return self.model
        
        model = UnstableDiffusion(**self.config["model_args"])
        return model.to(self.device)

    def infer_single_sample(self, images: torch.Tensor, segs: torch.tensor, datapoints: Datapoint) -> None:
        """
        image: single sample batch which has gone through the augmentations

        handle the result in fields
        """

        segs = segs/segs.max()
        segs = segs[:, 0:1, ...]
        # plt.imshow(seg[0].permute(1, 2, 0))
        # plt.show()
        # print("Mask")
        # print(seg.min(), seg.max(), seg.dtype, seg.shape, np.unique(seg))

        images = images[:, 0:3, ...]
        images = images/images.max()
        # print("Image")
        # print(image.min(), image.max(), image.dtype, image.shape)


        image = images.to(self.device)
        seg = segs.to(self.device, non_blocking=True)

        image = nn.functional.interpolate(image, scale_factor=2, mode='bicubic')
        seg = nn.functional.interpolate(seg, scale_factor=2, mode='nearest')

        save_path = self.save_path
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        # while os.path.exists(save_path):
        #     save_path = f"{self.lookup_root}/{input('Please enter a new save path: ')}"

        # os.mkdir(save_path)

        self.model.eval()
        with torch.no_grad():
            in_shape = list(self.config["target_size"])
            xt_im, xt_seg = self.progressive_denoise(image.shape[0], in_shape, model=self.model, image=image, seg=seg)
            
            print("Waring: not unbitifying. Only good for binary")
            xt_seg = xt_seg.cpu().permute(0, 2, 3, 1).numpy().astype(float)
            xt_seg -= xt_seg.min()
            xt_seg /= xt_seg.max()
            xt_seg = xt_seg.round() * 255
            xt_seg = xt_seg.astype(np.uint8)
            xt_im = xt_im.cpu().permute(0, 2, 3, 1).numpy()
            xt_im -= xt_im.min()
            xt_im *= 255 / xt_im.max()
            xt_im = xt_im.astype(np.uint8)
            for i in range(xt_im.shape[0]):
                cv2.imwrite(f"{save_path}/{datapoints[i].im_path.split('/')[-1]}", cv2.cvtColor(xt_im[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{save_path}/{datapoints[i].im_path.split('/')[-1].split('.')[0]}_seg.png", xt_seg[i])
                self.entries.append({
                    'IID': i,
                    'GID': 1,
                    'Image': f"{'/'.join(save_path.split('/')[-2:])}/{datapoints[i].im_path.split('/')[-1]}",
                    'Mask':  f"{'/'.join(save_path.split('/')[-2:])}/{datapoints[i].im_path.split('/')[-1].split('.')[0]}_seg.png",
                    'Label': 1
                })

        # xt_im = xt_im.detach().cpu()
                    
        # xt_im = xt_im.cpu()[0].permute(1, 2, 0).numpy()
        # xt_seg = bitmask_to_label(np.round(xt_seg.cpu()[0].permute(1, 2, 0).numpy()))
        # xt_seg = xt_seg.round()[0].cpu().permute(1, 2, 0).numpy()
        # return xt_im, xt_seg

    def progressive_denoise(self, batch_size, in_shape, model, image, seg):
        if model is None:
            model = self.model
        xt_im = torch.randn(
            (
                batch_size,
                self.config["model_args"]["im_channels"],
                *in_shape,
            )
        )
        xt_seg = torch.randn(
           (
               batch_size,
               self.config["model_args"]["seg_channels"],
               *in_shape,
           )
        )
        xt_im = xt_im.to(self.device)
        xt_seg = xt_seg.to(self.device)
        
        skip = self.timesteps // self.infer_timesteps
        seq = range(self.timesteps-1, -1, -skip)
        for t in tqdm(seq, desc="Running Inference"):
            time_tensor = (torch.ones(xt_im.shape[0]) * t).to(xt_im.device).long()
            t_n = t - skip if t !=0 else -1
            noise_prediction_im, noise_prediciton_seg = model(
                xt_im, xt_seg, torch.cat([image, seg], dim=1), time_tensor
            )
            xt_im, xt_seg = self.forward_diffuser.inference_call(
                xt_im,
                xt_seg,
                noise_prediction_im,
                noise_prediciton_seg,
                t,
                t_n,
                ddim=self.ddim,
            )            
        return xt_im, xt_seg

    def pre_infer(self) -> str:
        """
        Returns the output directory, and creates dataloader
        """
        self.model = self._get_model().cpu()
        checkpoint = torch.load(
            f"{self.lookup_root}/{self.weights}.pth"
        )["weights"]
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        print("!!!!!!Save path:", self.output_name)
        if self.output_name is None:
            return super().pre_infer()
        else:
            save_path = f'{self.lookup_root}/super_resolved_newest/{self.output_name}'
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path)
            return save_path, self.get_dataloader(self.batch)

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        columns = ['IID', 'GID', 'Image', 'Mask', 'Label']
        df = pd.DataFrame(self.entries, columns=columns)
        df.to_csv(f'{self.save_path}/generated.csv', index=False)
