import cProfile

import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from model import *
from dataset import *
from utils import *

from timeit import default_timer as timer
from datetime import datetime
import pandas as pd
import numpy as np
import shutil


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.image_size = option.image_size
        self.x_ranges = option.x_ranges
        self.y_ranges = option.y_ranges
        self.padding = option.padding
        self.fast_load = option.fast_load
        self.half = False
        self.dn_max = option.dn_max
        self.dataset_name = option.dataset
        self.pin_memory = option.pin_memory
        self.enable_transform = option.enable_transform
        self.scheduler_gamma = option.scheduler_gamma

        self.bands_cnt = option.bands_cnt
        self.bands = option.bands

        if self.bands:
            assert len(self.bands) == self.bands_cnt

        self.patch_size = option.patch_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_d = self.train_dir / 'discriminator.pth'

        self.logger = get_logger()
        self.logger.info('Model initialization')

        self.generator = AmsStf(num_in_ch=self.bands_cnt, num_out_ch=self.bands_cnt)
        self.discriminator = NLayerDiscriminator(input_nc=2 * self.bands_cnt, use_sigmoid=True, getIntermFeat=True)

        if self.half:
            self.generator.half()
            self.discriminator.half()

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids)

        self.weight_gan_loss = option.weight_gan_loss

        self.criterion = ReconstructionLoss(weight_content_loss=option.weight_content_loss,
                                            weight_pixel_loss=option.weight_pixel_loss,
                                            weight_spectral_loss=option.weight_spectral_loss,
                                            weight_vision_loss=option.weight_vision_loss)

        self.g_loss = GANLoss().to(self.device)
        self.d_loss = GANLoss().to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr, weight_decay=option.l2_alpha)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=option.d_lr, weight_decay=option.d_l2_alpha)

        self.sampler_train_set = None
        self.sampler_val_set = None

        if option.print_module:
            input_data = [[
                torch.randn(option.batch_size, self.bands_cnt, self.patch_size, self.patch_size),
                torch.randn(option.batch_size, self.bands_cnt, self.patch_size, self.patch_size),
                torch.randn(option.batch_size, self.bands_cnt, self.patch_size, self.patch_size),
                torch.randn(option.batch_size, self.bands_cnt, self.patch_size, self.patch_size),
                torch.randn(option.batch_size, self.bands_cnt, self.patch_size, self.patch_size),
            ]]
            summary(self.generator, input_data=input_data)

            n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
            self.logger.info(f'There are {n_params} trainable parameters for generator.')

            n_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
            self.logger.info(f'There are {n_params} trainable parameters for discriminator.')

    def train_on_epoch(self, n_epoch, data_loader):
        self.generator.train()

        self.discriminator.train()

        epg_loss = AverageMeter()
        epd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()

            data = [im.to(self.device) for im in data]
            if self.half:
                data = [im.half() for im in data]

            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)

            self.generator.zero_grad()

            ############################
            # (1) Update D network
            ###########################
            self.discriminator.zero_grad()

            pred_fake = self.discriminator(torch.cat((prediction.detach(), inputs[-1]), 1))
            pred_real1 = self.discriminator(torch.cat((target, inputs[-1]), 1))

            d_loss = (self.d_loss(pred_fake, False) +
                        self.d_loss(pred_real1, True)) * 0.5

            d_loss.backward()
            self.d_optimizer.step()
            epd_loss.update(d_loss.item())

            ############################
            # (2) Update G network
            ###########################
            a_loss = self.criterion(prediction, target)

            gan_loss = self.g_loss(self.discriminator(torch.cat((prediction, inputs[-1]), 1)), True)
            a_loss = a_loss + self.weight_gan_loss * gan_loss

            a_loss.backward()
            self.g_optimizer.step()
            epg_loss.update(a_loss.item())

            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()

            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                                f'A-Loss: {a_loss.item():.6f} - '
                                f'D-Loss: {d_loss.item():.6f} - '
                                f'MSE: {mse:.6f} - '
                                f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')

        t_start = timer()
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        t_end = timer()
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()} Time: {t_end - t_start}s save generator checkpoint')

        t_start = timer()
        save_checkpoint(self.discriminator, self.d_optimizer, self.last_d)
        t_end = timer()
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()} Time: {t_end - t_start}s '
                            f'save discriminator checkpoint')

        return epg_loss.avg, epd_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, n_epoch, data_loader):
        self.generator.eval()
        self.discriminator.eval()

        epoch_error = AverageMeter()
        epoch_loss = AverageMeter()

        epoch_sobel_loss = AverageMeter()
        epoch_feature_loss = AverageMeter()
        epoch_spectral_loss = AverageMeter()
        epoch_vision_loss = AverageMeter()
        epoch_gan_loss = AverageMeter()

        should_record_img = False
        if n_epoch % 4 == 0:
            should_record_img = True

        t_start = timer()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            if self.half:
                data = [im.half() for im in data]

            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)

            test_loss, loss_details = self.criterion(prediction, target, output_details=True)

            epoch_sobel_loss.update(loss_details["sobel_loss"])
            epoch_feature_loss.update(loss_details["feature_loss"])
            epoch_spectral_loss.update(loss_details["spectral_loss"])
            epoch_vision_loss.update(loss_details["vision_loss"])

            gan_loss = self.g_loss(self.discriminator(torch.cat((prediction, inputs[-1]), 1)), True)
            epoch_gan_loss.update(gan_loss)

            test_loss = test_loss + 1e-2 * gan_loss

            epoch_loss.update(test_loss)
            
            error = F.mse_loss(prediction, target).item()
            epoch_error.update(error)

        t_end = timer()
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()} Time: {t_end - t_start}s Val test done')

        return epoch_error.avg, epoch_loss.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              epochs=30, num_workers=0, resume=True):
        last_epoch = -1
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_d, self.discriminator, optimizer=self.d_optimizer)

        start_epoch = last_epoch + 1
        least_error = float('inf')

        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, self.patch_size, patch_stride, mode=Mode.TRAINING,
                             x_ranges=self.x_ranges, y_ranges=self.y_ranges, fast_load=self.fast_load,
                             half=self.half, dataset_name=self.dataset_name, dn_max=self.dn_max,
                             enable_transform=self.enable_transform, bands=self.bands)
        val_set = PatchSet(val_dir, self.image_size, self.patch_size, patch_stride, mode=Mode.VALIDATION,
                           x_ranges=self.x_ranges, y_ranges=self.y_ranges, fast_load=self.fast_load,
                           half=self.half, dataset_name=self.dataset_name, dn_max=self.dn_max,
                           enable_transform=self.enable_transform, bands=self.bands)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, prefetch_factor=2,
                                  persistent_workers=True, sampler=self.sampler_train_set,
                                  num_workers=num_workers, drop_last=True, pin_memory=self.pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, prefetch_factor=2, persistent_workers=True,
                                sampler=self.sampler_val_set,
                                num_workers=num_workers, pin_memory=self.pin_memory)

        self.logger.info('Training...')
        g_scheduler = ExponentialLR(self.g_optimizer, self.scheduler_gamma)

        d_scheduler = ExponentialLR(self.d_optimizer, self.scheduler_gamma)

        profile_done = False

        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for Generator: {self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: {self.d_optimizer.param_groups[0]['lr']}")

            train_g_loss, train_d_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error, val_loss = self.test_on_epoch(epoch, val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_d_loss', 'train_error', 'val_error', 'val_loss']
            csv_values = [epoch, train_g_loss, train_d_loss, train_g_error, val_error, val_loss]
            log_csv(self.history, csv_values, header=csv_header)
            g_scheduler.step()
            d_scheduler.step()

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        load_checkpoint(self.best, self.generator)
        self.generator.eval()
        patch_size = make_tuple(patch_size)
        padding = make_tuple(self.padding)
        self.logger.info('Predicting...')

        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        image_paths = [get_pair_path(d, Mode.PREDICTION, self.dataset_name) for d in image_dirs]

        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        test_set = PatchSet(test_dir, self.image_size, patch_size, mode=Mode.PREDICTION,
                            padding=padding, remove_minus=True, fast_load=self.fast_load,
                            dataset_name=self.dataset_name, dn_max=self.dn_max, bands=self.bands)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers, pin_memory=self.pin_memory)

        im_count = 0
        patches = []
        weights = []
        for data in test_loader:
            inputs = [im.to(self.device) for im in data]
            name = image_paths[im_count][-1].name
            if len(patches) == 0:
                t_start = timer()
                self.logger.info(f'Predict on image {name}')
            prediction = self.generator(inputs)
            prediction = prediction.squeeze_().cpu().numpy()
            prediction = (prediction * self.dn_max).astype(np.int16)
            prediction = prediction[:, padding[0]:(padding[0] + patch_size[0]),
                                    padding[0]:(padding[1] + patch_size[1])]

            patches.append(prediction)

            if len(patches) == n_blocks:
                result = np.empty((self.bands_cnt, *self.image_size), dtype=np.int16)

                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                               col_start: col_start + patch_size[0],
                               row_start: row_start + patch_size[1], ] = patches[block_count]

                        block_count += 1

                patches.clear()
                weights.clear()

                prototype = str(image_paths[im_count][0])
                save_array_as_tif(result, self.test_dir / name, prototype=prototype)

                im_count += 1
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')
