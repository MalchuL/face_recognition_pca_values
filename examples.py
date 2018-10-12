import torch
import models
from Datasetter import HackatonDataset
import torchvision.datasets
from torch.utils.data import DataLoader


def get_batch(batch_size=10):


    return torch.randn(batch_size, 3, 450, 450), torch.randn(batch_size, 199)


if __name__ == '__main__':
    dset = HackatonDataset('./datasets/', './datasets/train/', '.jpg')
    train_loader = DataLoader(dset,
                              batch_size=10,
                              shuffle=False,
                              num_workers=2
                              # pin_memory=True # CUDA only
                              )
    data = dset.get_train_batch(10)
    print(data[0].size(),data[1].size())

    """for batch in enumerate(train_loader):

    batch = get_batch(10, train_loader)

    module = models.ResNetDepth(num_channels=3,layers=[1,1,1,1],num_elements=160)
    with torch.no_grad():
        module, batch = module.cuda(), batch.cuda()
        output = module(batch)
        print(output, output.size)"""

