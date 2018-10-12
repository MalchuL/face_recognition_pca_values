import pandas as pd
import PIL.Image
import numpy as np
import torchvision.transforms as tr
from torchvision.transforms import *
import torch.utils.data




class HackatonDataset(torch.utils.data.Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, names_path, img_path, img_ext):
        #ToDo check for names_path end with .txt else load all files from names_path
        tmp_df = list(map(lambda x:x.replace('\n',''),open(names_path).readlines()))


#"Some images referenced in the CSV file were not found"

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform=tr.Compose([tr.Resize((450, 450)), tr.RandomAffine(25), tr.ColorJitter(0.3, 0.5, 0.1), tr.RandomHorizontalFlip(), tr.ToTensor()])
        self.X_train = tmp_df


    def get_size(self):
        return len(self.X_train)


    def __getitem__(self, index):
        img = PIL.Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        csv = pd.read_csv(self.img_path + self.X_train[index]+'.csv', header=None)
        csv = np.hstack(csv.values)
        if self.transform is not None:
            img = self.transform(img)
        return img, csv

    def get_train_batch(self, batch_size):
        length = len(self.X_train)
        array=np.random.choice(length, batch_size)
        X, Y=[],[]
        for element in array:
            x, y= self[element]
            y = torch.from_numpy(y)
            print(type(y))
            X.append(x)
            Y.append(y)
        X = torch.stack(X,0)
        Y = torch.stack(Y,0)
        return X,Y

    def get_ordered_batch(self, offset, batch_size):
        array=range((offset)*batch_size, (offset+1)*batch_size)
        X, Y=[],[]
        for element in array:
            x, y= self[element]
            y = torch.from_numpy(y)
            print(type(y))
            X.append(x)
            Y.append(y)
        X = torch.stack(X,0)
        Y = torch.stack(Y,0)
        return X,Y

    def __len__(self):
        return len(self.X_train.index)

