import os
from torch.utils.data import Dataset
from PIL import Image

class ChineseMNISTDataset(Dataset):
    def __init__(self, df, img_folder_path, transform=None):
        super().__init__()
        self.df = df
        self.img_folder_path = img_folder_path
        self.transform = transform
  
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["img_path"])
        image = self.transform(image)
        return image, row["code"] - 1
  
    def __len__(self):
        return len(self.df)
