import pandas as pd
import PIL as Image
import os
import numpy as np
import torch
import torchvision.transforms as tr
import torch.utils.data.Dataset



class KaggleAmazonDataset(torch.utils.data.Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=tr.Compose([tr.Resize((450, 450)), tr.RandomAffine(25), tr.ColorJitter(0.3, 0.5, 0.1), tr.RandomHorizontalFlip(), tr.ToTensor()])):

        tmp_df = pd.read_csv(csv_path+'train.csv')
       # assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
#"Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform
        self.X_train = tmp_df
        #self.y_train = pd.read_csv('./datasets/images/0001.csv', header=None)



    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)

        img = img.convert('RGB')
        csv = pd.read_csv(self.csv_path+self.X_train[index]+'.csv', header=None)
        csv = np.hstack(csv.values)
        if self.transform is not None:
            img = self.transform(img)
        return img, csv

    def __len__(self):
        return len(self.X_train.index)

