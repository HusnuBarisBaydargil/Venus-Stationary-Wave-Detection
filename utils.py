import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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

def load_data(dataset_path, batch_size):
    train_data = ImagePaths(dataset_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=10, shuffle=False)
    return train_loader
