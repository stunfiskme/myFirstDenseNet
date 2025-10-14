import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Custom IP102 dataset
class IP102Dataset(Dataset):
    def __init__(self, split_file, image_root, transform=None):
        self.samples = []
        self.image_root = image_root
        self.transform = transform

        with open(split_file, 'r') as f:
            for line in f:
                image_rel_path, label = line.strip().split()
                self.samples.append((image_rel_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_rel_path, label = self.samples[idx]
        image_path = os.path.join(self.image_root, str(label), image_rel_path)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

#
gaussian_blur = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
color_jitter = transforms.ColorJitter(
            brightness=0.4,   # ±x% brightness
            contrast=0.4,     # ±x% contrast
            saturation=0.4,   # ±x%saturation
            hue=0.2          # ±x% hue shift
)
# Transform: resize and convert to tensor
# Compose: do everything below in order as listed
transform = transforms.Compose([
    transforms.Resize((224,224)),        
    transforms.ToTensor(),
    transforms.Normalize(                          # maps [0,1]→[-1,1]
        mean=(0.5, 0.5, 0.5), 
        std=(0.5, 0.5, 0.5)
    )
])

train_transform = transforms.Compose([
    transforms.Resize((224,224)),        
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([gaussian_blur], 0.5),
    transforms.RandomApply([color_jitter], p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
    transforms.Normalize(                          # maps [0,1]→[-1,1]
        mean=(0.5, 0.5, 0.5), 
        std=(0.5, 0.5, 0.5)
    )
])


# Dataset 
training_data = IP102Dataset(
    split_file="/content/ip02-dataset/train.txt",
    image_root="/content/ip02-dataset/classification/train",
    transform=train_transform
)

validation_data = IP102Dataset(
    split_file="/content/ip02-dataset/val.txt",
    image_root="/content/ip02-dataset/classification/val",
    transform=train_transform
)

test_data = IP102Dataset(
    split_file="/content/ip02-dataset/test.txt",
    image_root="/content/ip02-dataset/classification/test",
    transform=transform
)

# Dataloaders
train_dataloader = DataLoader(
    training_data,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=2)

validation_dataloader = DataLoader(
    validation_data,
    batch_size=16,
    shuffle=True,
    pin_memory=True,
    num_workers=2)

test_dataloader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False,
    pin_memory=True,
    num_workers=2
)