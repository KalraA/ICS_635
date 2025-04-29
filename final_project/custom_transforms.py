import random
import numpy as np
import torch
from transformers import AutoImageProcessor
from torchvision import transforms
from torchvision.transforms import functional as F

class DinoPostProcess:
    def __init__(self, model_name):
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.image_processor.do_resize=False
        self.image_processor.do_center_crop=False        
        print(self.image_processor)
        
    def __call__(self, image, target):
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs['pixel_values'][0], target
        


class Load:
    def __init__(self, colormap):
        self.colormap = np.array([v for k, v in sorted(colormap.items())])

    def __call__(self, image, target):
        image = transforms.functional.pil_to_tensor(image)
        target = np.array(target.convert("RGB"))
        target = (self.colormap[None, None] == target[:, :, None]).sum(axis=-1).argmax(axis=-1)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target[None]