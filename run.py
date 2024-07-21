import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from experiment import Experiment
import faulthandler

faulthandler.enable()

# For model
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion model')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the initial learning rate of generator')
parser.add_argument('--d-lr', type=float, default=1e-4,
                    help='the initial learning rate of discriminator')
parser.add_argument('--l2-alpha', type=float, default=1e-2,
                    help='the weight decay for L2 regularization of generator')
parser.add_argument('--d-l2-alpha', type=float, default=1e-2,
                    help='the weight decay for L2 regularization of discriminator')
parser.add_argument('--scheduler-gamma', type=float, default=0.99,
                    help='the gamma of scheduler')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--batch-size-per-gpu', type=int, default=6,
                    help='the batch size of per gpu for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('.'),
                    help='the output directory')
parser.add_argument('--seed', type=int, default=1, help='the seed for random')

# For dataset
parser.add_argument('--dataset', type=str, help='the type of dataset')
parser.add_argument('--train_dir', type=Path, help='the training data directory')
parser.add_argument('--val_dir', type=Path, help='the validation data directory')
parser.add_argument('--test_dir', type=Path, help='the test data directory')
parser.add_argument('--image_size', type=int, nargs='+', required=True,
                    help='the size of the coarse image (width, height)')

parser.add_argument('--bands-cnt', type=int, default=6,
                    help='count of bands')
parser.add_argument('--bands', type=int, nargs='+', help='Bands numbers for train and prediction')

parser.add_argument('--patch-size', type=int, default=128,
                    help='patch size for training')
parser.add_argument('--x_ranges', type=int, nargs='+',
                    help='the x ranges of image for cutting edges when training')
parser.add_argument('--y_ranges', type=int, nargs='+',
                    help='the y ranges of image for cutting edges when training')
parser.add_argument('--patch_stride', type=int, nargs='+',
                    help='the coarse patch stride for image division when training')
parser.add_argument('--dn-max', type=int, default=10000,
                    help='the max number of DN value of the image')
parser.add_argument('--enable-transform', action='store_true', help='Enable randomly transforms')
parser.add_argument('--pin-memory', action='store_true', help='Enable pinned memory')

parser.add_argument('--test_patch', type=int, nargs='+',
                    help='the coarse image patch size when testing')
parser.add_argument('--padding', type=int, default=0, help='padding of patch image when testing')

parser.add_argument('--weight-content-loss', type=float, default=1.0, help='the weight of content loss')
parser.add_argument('--weight-pixel-loss', type=float, default=1.0, help='the weight of pixel loss')
parser.add_argument('--weight-spectral-loss', type=float, default=1.0, help='the weight of spectral loss')
parser.add_argument('--weight-vision-loss', type=float, default=1.0, help='the weight of vision loss')
parser.add_argument('--weight-gan-loss', type=float, default=1e-2, help='the weight of vision loss')

parser.add_argument('--print-module', action='store_true', help='print module info')

parser.add_argument('--fast-load', action='store_true', help='Cache image in memory')

opt = parser.parse_args()

torch.manual_seed(opt.seed)

if opt.cuda and not torch.cuda.is_available():
    opt.cuda = False
else:
    cudnn.benchmark = True
    cudnn.deterministic = True

if opt.cuda:
    print("CUDA: enabled")
else:
    print("CUDA: disabled or not supported!")

opt.batch_size = opt.ngpu * opt.batch_size_per_gpu


if __name__ == '__main__':
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir, opt.patch_stride,
                         opt.batch_size, epochs=opt.epochs,
                         num_workers=opt.num_workers)
    experiment.test(opt.test_dir, opt.test_patch, num_workers=opt.num_workers)

