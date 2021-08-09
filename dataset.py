import glob
import os 
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageFile
import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        if label>=3:
            label=label-1
        return {"images":image,"labels":torch.tensor(label),"image_names":image_name.split("/")[-1]}

    def __len__(self):
        return len(self.files)

if __name__=="__main__":
    transforms_ = [ transforms.Resize(400),
                    transforms.CenterCrop((320,320)), 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    a=ImageDataset("../classification/trainSet","../classification/train.json",transforms_)
    b=a.__getitem__(1)
    print(b)
    dataloader = DataLoader(a,batch_size=2, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch)
        i+=1
        if i==10:
            break
