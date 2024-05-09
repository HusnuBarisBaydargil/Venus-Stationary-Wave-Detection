import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ImagePaths(Dataset):
    def __init__(self, path):
        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0)
        image = image[np.newaxis, :, :]

        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example

def load_data(dataset_path, batch_size, is_distributed=False, rank=0, num_replicas=1):
    train_data = ImagePaths(dataset_path)
    if is_distributed:
        sampler = DistributedSampler(train_data, num_replicas=num_replicas, rank=rank, shuffle=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=10, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)
    
    return train_loader, sampler if is_distributed else train_loader