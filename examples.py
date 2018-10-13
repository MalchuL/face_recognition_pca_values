import argparse
import fnmatch
import os
import torch
import PIL.Image
import models
import torchvision.transforms as tr
from Datasetter import HackatonDataset
import train
import numpy as np

def pred(path, dest, trainer):
    dirs = os.listdir(path)
    preprocess = tr.Compose([tr.Resize((450, 450)), tr.ToTensor()])
    for file in dirs:
        if fnmatch.fnmatch(file, '*.jpg'):
            img = PIL.Image.open(path + file)
            x = preprocess(img.convert('RGB'))
            x = torch.stack([x], 0).type(torch.FloatTensor)
            x.requires_grad=False
            res = trainer.eval(x)
            np.savetxt(dest + file.replace('jpg', 'csv'), res.numpy(), delimiter='\n')
            
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('--dest_path_pred', default='./datasets/dest/', type=str,
                        help='path to prediction results folder')
    parser.add_argument('--img_path_pred', default='./datasets/test/', type=str,
                        help='path to prediction data folder')
    parser.add_argument('--names_path_train', default='./datasets/train.txt', type=str,
                        help='path to train files with names in folder')
    parser.add_argument('--img_path_train', default='./datasets/train/', type=str,
                        help='path to train data folder')
    parser.add_argument('--names_path_test', default='./datasets/test.txt', type=str,
                        help='path to test files with names in folder')
    parser.add_argument('--img_path_test', default='./datasets/train/', type=str,
                        help='path to test data folder')
    parser.add_argument('--checkpoint_path', default='./data.ckpt', type=str,
                        help='path to checkpoint data')
    parser.add_argument('--mode', default=0, type=int,
                        help='1 - train, 0 - eval')
    parser.add_argument('--gpu', default=False, type=bool,
                        help='set is gpu used')

    parser.add_argument('--batch_size', default='10', type=int,
                        help='batches')

    parser.add_argument('--path_to_normalizer', default='./scaler.obj', type=str,
                        help='loss which need to save')

    parser.add_argument('--global_error', default='10000', type=float,
                        help='loss which need to save')

    FLAGS = parser.parse_args()

    dset_train = HackatonDataset(FLAGS.names_path_train, FLAGS.img_path_train, '.jpg')
    dset_test = HackatonDataset(FLAGS.names_path_test, FLAGS.img_path_test, '.jpg')

    train_data_size = dset_train.get_size()
    test_data_size = dset_test.get_size()

    use_cuda = FLAGS.gpu
    print(FLAGS)
    trainer = train.Trainer(use_cuda,dset_train.get_train_batch, dset_test.get_ordered_batch, checkpoint_path=FLAGS.checkpoint_path,global_loss=FLAGS.global_error)
    if FLAGS.mode == 1:
        trainer.model.train()
        trainer.train(1000, FLAGS.batch_size, train_data_size, test_data_size)
    elif FLAGS.mode == 0:
        pred(FLAGS.img_path_pred, FLAGS.dest_path_pred, trainer)
