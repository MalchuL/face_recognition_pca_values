import torch
import models
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def prepare_scaler(name_file, csv_dir):
    tmp_df = np.genfromtxt(name_file, delimiter='\n', dtype=str)
    files = []
    for name in tmp_df:
        files.append(np.genfromtxt(csv_dir + name + '.csv', delimiter = '\n'))
    scaler = StandardScaler()
    scaler.fit(files)
    return scaler

def dump_object(obj, file_name):
    file = open(file_name, 'wb')
    pickle.dump(obj, file)
    file.close()

def restore_object(file_name):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close()

def get_batch(batch_size=10):
    return torch.randn(batch_size,3, 450, 450),torch.randn(batch_size,199)

if __name__ == '__main__':
    batch = get_batch(10)
    scaler = prepare_scaler('train.csv', './train/')
    scaler.transform(batch)
    module = models.ResNetDepth(num_channels=3,layers=[1,1,1,1],num_elements=160)
    with torch.no_grad():
        module, batch = module.cuda(), batch.cuda()
        output = module(batch)
        print(output, output.size)
