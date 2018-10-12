import torch
import torch.nn as nn
import torch.functional as F

class FaceReconstruction(nn.Module):
    def __init__(self, image_size, output_size, n_channels=3):
        if(isinstance(image_size,tuple)):
            self.image_size = image_size
        else:
            self.image_size = (image_size,image_size)
        self.output_size = output_size
        self.n_channels = n_channels




if __name__ == '__main__':
    pass