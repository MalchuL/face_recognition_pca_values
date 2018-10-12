import argparse

import torch
import models
from Datasetter import HackatonDataset
import torchvision.datasets
from torch.utils.data import DataLoader
import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('--names_path_train', default='./datasets/train.txt', type=str,
                        help='path to train files with names in folder')
    parser.add_argument('--img_path_train', default='./datasets/', type=str,
                        help='path to train data folder')
    parser.add_argument('--names_path_test', default='./datasets/test.txt', type=str,
                        help='path to test files with names in folder')
    parser.add_argument('--img_path_test', default='./datasets/', type=str,
                        help='path to test data folder')
    parser.add_argument('--checkpoint_path', default='TestImages/output.jpg', type=str,
                        help='path to checkpoint data')
    parser.add_argument('--mode', default=1, type=int,
                        help='1 - train, 0 - eval')
    parser.add_argument('--gpu', default='True', type=bool,
                        help='set is gpu used')

    parser.add_argument('--batch_size', default='10', type=int,
                        help='batches')

    FLAGS = parser.parse_args()

    module = models.ResNetDepth(num_channels=3, layers=[2, 3, 4, 2], num_elements=199)

    dset_train = HackatonDataset(FLAGS.names_path_train, FLAGS.img_path_train, '.jpg')
    dset_test = HackatonDataset(FLAGS.names_path_test, FLAGS.img_path_test, '.jpg')

    train_data_size = dset_train.get_size()
    test_data_size = dset_test.get_size()

    use_cuda = FLAGS.gpu

    trainer = train.Trainer(use_cuda,dset_train.get_train_batch, dset_test.get_ordered_batch, checkpoint_path=FLAGS.checkpoint_path)
    if FLAGS.mode == 1:
        trainer.model.train()
        trainer.train(1000, FLAGS.batch_size, train_data_size, test_data_size)
    elif FLAGS.mode == 0:
        trainer.model.eval()
