import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial

from torch import nn
from torchvision import transforms
from torch.utils import data as data
from pathlib import Path

# from .realesrgan import RealESRGAN_degradation
# from myutils.img_util import convert_image_to_fn
# from myutils.misc import exists

class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="datasets/pngtxt", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        self.size = image_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # Define the root directory and sub-directory paths
        self.root_dir = Path('/content/oxford-102-flower-dataset/102 flower/flowers/train')
        if not self.root_dir.exists():
            raise ValueError("Root directory doesn't exist.")

        self.image_paths = []
        self.prompts = []

        # Iterate through each class folder to collect image paths and corresponding prompts
        for class_folder in self.root_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                for image_file in class_folder.glob('*.jpg'):  # Adjust the extension as necessary
                    self.image_paths.append(image_file)
                    self.prompts.append(f"a photo of a {'<'+{flowers[class_name]}+'>'}")

        self.num_images = len(self.image_paths)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        example = {}

        # Load image
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = self.prompts[index]

        example['instance_images'] = self.image_transforms(image)
        example['instance_prompt_ids'] = self.tokenizer(
            prompt,
            padding='do_not_pad',
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids

        return example