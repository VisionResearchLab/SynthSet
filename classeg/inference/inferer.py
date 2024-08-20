import glob
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from json_torch_models.model_factory import ModelFactory
from torch.utils.data import DataLoader
from tqdm import tqdm

from classeg.dataloading.datapoint import Datapoint
from classeg.dataloading.dataset import PipelineDataset
from classeg.utils.constants import *
from classeg.utils.utils import batch_collate_fn
from classeg.utils.utils import read_json


class Inferer:
    def __init__(
            self,
            dataset_name: str,
            fold: int,
            name: str,
            weights: str,
            input_root: str,
            late_model_instantiation=False,
            **kwargs
    ):
        """
        Inferer for pipeline.
        :param dataset_name: The dataset_name used for training
        :param fold: The fold to run inference with
        :param name: The name of the experiment with the trained weights and config.
        :param weights: The name of the weights to load.
        """
        self.save_path = None
        self.train_loader = None
        # ^^^ both are late initialized
        self.dataset_name = dataset_name
        self.fold = fold
        self.input_root = input_root
        self.lookup_root = (
            f"{RESULTS_ROOT}/{self.dataset_name}/fold_{fold}/{name}"
        )
        self.config = read_json(f"{self.lookup_root}/config.json")
        self.weights = weights
        self.device = "cuda"
        if not late_model_instantiation:
            self.model = self.get_model()
        else:
            self.model = None
        assert os.path.exists(self.lookup_root)
        #assert torch.cuda.is_available(), "No gpu available."

    def get_model(self) -> nn.Module:
        """
        Loads the model and weights.
        :return:
        """
        checkpoint = torch.load(
            f"{self.lookup_root}/{self.weights}.pth"
        )
        model = ModelFactory(
            json_path=f"{self.lookup_root}/model.json",
            lookup_packages=["classeg.models.autoencoder", "classeg.models.unet"]
        ).get_model()
        model.load_state_dict(checkpoint["weights"])
        return model.to(self.device)

    def pre_infer(self) -> Tuple[str, DataLoader]:
        """
        Returns the output directory, and creates dataloader
        """
        save_path = f'{self.lookup_root}/inference'
        while os.path.exists(save_path):
            new_name = input("To avoid overwriting old inference, enter a name for the output folder: ")
            save_path = '/'.join(save_path.split("/")[0:-1])
            save_path = f"{save_path}/{new_name}"

        os.makedirs(save_path)
        return save_path, self.get_dataloader()

    @abstractmethod
    def get_augmentations(self):
        """
        Return the augmentations to apply to each sample

        Remember that you can query the train set mean/std for standardization at:
            PREPROCESSED_ROOT/self.dataset_name/fold_{self.fold}/mean_std.json
        """
        ...

    @abstractmethod
    def infer_single_sample(self, image: torch.Tensor, datapoint: Datapoint) -> None:
        """
        image: single sample batch which has gone through the augmentations

        handle the result in fields
        """
        ...

    def post_infer(self):
        """
        Here, inference has run on every sample.

        Take advantage of what you saved in infer_single_epoch to write something meaningful
        (or not, if you did something else)
        """
        ...

    def get_dataloader(self, batch_size=256) -> DataLoader:
        datapoints = self._get_datapoints()
        dataset = PipelineDataset(datapoints, self.dataset_name, transforms=self.get_augmentations())
        return DataLoader(
            dataset=dataset,
            pin_memory=True,
            num_workers=self.config["processes"],
            batch_size=batch_size,
            collate_fn=batch_collate_fn
        )

    def infer(self):
        """
        Creates the self values for save path and loader
        """
        self.save_path, self.train_loader = self.pre_infer()
        self.model.eval()
        with torch.no_grad():
            for image, label, datapoints in tqdm(self.train_loader, desc="Running inference"):
                if label is not None:
                    self.infer_single_sample(image, label, datapoints)
                else:        
                    self.infer_single_sample(image, datapoints)
        self.post_infer()

    def _get_datapoints(self):
        if os.path.isdir(glob.glob(f"{self.input_root}/*")[0]):
            paths = [x for x in glob.glob(f"{self.input_root}/images/*") if not os.path.isdir(x)]
            datapoints = [Datapoint(x, x.replace("/images", "/masks")) for x in paths]
        else:
            paths = [x for x in glob.glob(f"{self.input_root}/*") if not os.path.isdir(x)]
            datapoints = [Datapoint(x, None) for x in paths]

        print(f"Found {len(paths)} images to infer on.")
        if len(paths) == 0:
            raise SystemExit
        return datapoints
