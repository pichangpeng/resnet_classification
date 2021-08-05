import glob
import os 
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_path,json_path,transforms_):
        self.transforms=transforms.Compose(transforms_)
        self.files = sorted(glob.glob(image_path + '/*.*'))
        with open(json_path, 'r') as f:
            self.image_lab_dict = json.load(f)

    def __getitem__(self, index):
        image_name=self.files[index % len(self.files)]
        image=self.transforms(Image.open(image_name).convert('RGB'))
        label=self.image_lab_dict[image_name.split("/")[-1]]
        return {"images":image,"labels":label}

    def __len__(self):
        return len(self.files)