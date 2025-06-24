import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, images_folder, target_size=(256, 256), max_samples=2000):
        self.data = pd.read_csv(csv_file).head(max_samples)
        self.images_folder = images_folder
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Index"]
        img_path = os.path.join(self.images_folder, img_name)
        img = Image.open(img_path).convert("RGB").resize(self.target_size)
        img_tensor = torch.tensor(np.array(img).astype(np.float32).transpose(2, 0, 1)) / 127.5 - 1.0
        prompt = f"A chest x-ray with {self.data.iloc[idx]['Finding Labels']}"
        return {"image": img_tensor, "prompt": prompt}

def create_dataloader():
    csv_file = "data/sample_labels.csv"
    images_folder = "data/images"
    dataset = ChestXrayDataset(csv_file, images_folder)
    return DataLoader(dataset, batch_size=1, shuffle=True), {}