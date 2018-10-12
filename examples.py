import torch
import FaceReconstructionModule

def get_batch(batch_size=10):
    return torch.randn(3, 450, 450)

if __name__ == '__main__':
    batch = get_batch(10)
    module = FaceReconstructionModule(450, 160, 3)
    output = module(batch)
    print(output, output.size)