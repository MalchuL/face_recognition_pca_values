import torch
import models

def get_batch(batch_size=10):
    return torch.randn(3, 450, 450)

if __name__ == '__main__':
    batch = get_batch(10)
    module = models.ResNetDepth(num_channels=3,layers=[1,1,1,1],num_elements=160)
    output = module(batch)
    print(output, output.size)