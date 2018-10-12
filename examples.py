import argparse

import torch
import models
from Datasetter import HackatonDataset
import torchvision.datasets
from torch.utils.data import DataLoader
import train

def get_batch():
    return torch.ones(2,3,450,450)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

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
    parser.add_argument('--mode', default=1, type=int,
                        help='1 - train, 0 - eval')
    parser.add_argument('--gpu', default='True', type=bool,
                        help='set is gpu used')
    parser.add_argument('--test_aaaa', default='0', type=int,
                        help='not set in production')

    parser.add_argument('--batch_size', default='10', type=int,
                        help='batches')

    parser.add_argument('--path_to_normalizer', default='./scaler.obj', type=str,
                        help='loss which need to save')

    parser.add_argument('--global_error', default='10000', type=float,
                        help='loss which need to save')

    FLAGS = parser.parse_args()

    module = models.ResNetDepth(num_channels=3, layers=[2, 3, 4, 2], num_elements=199)

    dset_train = HackatonDataset(FLAGS.names_path_train, FLAGS.img_path_train, '.jpg')
    dset_test = HackatonDataset(FLAGS.names_path_test, FLAGS.img_path_test, '.jpg')

    train_data_size = dset_train.get_size()
    test_data_size = dset_test.get_size()

    use_cuda = FLAGS.gpu
    print(FLAGS)
    trainer = train.Trainer(use_cuda,dset_train.get_train_batch, dset_test.get_ordered_batch, checkpoint_path=FLAGS.checkpoint_path,global_loss=FLAGS.global_error)
    if FLAGS.mode == 1:
        trainer.model.train()
        trainer.skip_test=FLAGS.test_aaaa
        trainer.train(1000, FLAGS.batch_size, train_data_size, test_data_size)
    elif FLAGS.mode == 0:
        print(trainer.model(get_batch()))
