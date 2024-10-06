"""
The autoencoder model and the adversarial training was modified based on the Face-Aging-CAAE project
The original code can be found in: https://github.com/ZZUTK/Face-Aging-CAAE
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import *
import consts

import logging
import random
from collections import OrderedDict
import cv2
import imageio
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
import pdb


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    act_fn
                )
            )

        add_conv(self.conv_layers, 'e_conv_1', in_ch=1, out_ch=64, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', in_ch=64, out_ch=128, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', in_ch=128, out_ch=256, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', in_ch=256, out_ch=512, kernel=5, stride=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', in_ch=512, out_ch=1024, kernel=5, stride=2, act_fn=nn.ReLU())
        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ('e_fc_1', nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS)),
                    ('tanh_1', nn.Tanh())  # normalize to [-1, 1] range
                ]
            )
        )

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        out = out.flatten(1, -1)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dz_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dz_fc_%d' % (i + 1),
            nn.Sequential(
                nn.Linear(out_dim, 1),
            )
        )

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        return out


class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        in_dims = (1, 16, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                nn.Linear(128 * 8 * 8, 1024),
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1),
            )
        )

    def forward(self, imgs):
        out = imgs
        for i, conv_layer in enumerate(self.conv_layers, 1):
            out = conv_layer(out)
        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        mini_size = 4
        self.fc = nn.Sequential(
            nn.Linear(
                consts.NUM_Z_CHANNELS,
                consts.NUM_GEN_CHANNELS * mini_size ** 2
            ),
            nn.ReLU()
        )

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 4, 4), out_dims=(512, 8, 8), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 8, 8), out_dims=(256, 16, 16), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 16, 16), out_dims=(128, 32, 32), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 32, 32), out_dims=(64, 64, 64), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_5', in_dims=(64, 64, 64), out_dims=(32, 128, 128), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_6', in_dims=(32, 128, 128), out_dims=(16, 128, 128), kernel=5, stride=1, actf=nn.ReLU())
        add_deconv('g_deconv_7', in_dims=(16, 128, 128), out_dims=(1, 128, 128), kernel=1, stride=1, actf=nn.Tanh())

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 4, 4)

    def forward(self, z):
        out = z
        out = self.fc(out)
        out = self._decompress(out)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            out = deconv_layer(out)
        return out


class Net(object):
    def __init__(self):
        self.E = Encoder().cuda()
        self.Dz = DiscriminatorZ().cuda()
        self.Dimg = DiscriminatorImg().cuda()
        self.G = Generator().cuda()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = 'cuda'

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])

    def train_model(
            self,
            embryoid_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
            betas=(0.9, 0.999),
            valid_size=None,
            where_to_save=None,
            args=None
    ):
        sample_name_list = [file for file in os.listdir(os.path.join(os.getcwd(), args.input_data_path))
                            if os.path.isfile(os.path.join(os.getcwd(), args.input_data_path, file))]

        dataset = Embryoid_dataset(base_dir=args.input_data_path, sample_name_list=sample_name_list,
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]))

        valid_size = valid_size or batch_size
        valid_dataset, train_dataset = torch.utils.data.random_split(dataset, (valid_size, len(dataset)-valid_size))

        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        logging.basicConfig(filename=os.path.join(args.output_path,'log.txt'), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        for epoch in range(1, epochs + 1):

            losses = defaultdict(lambda: [])

            self.train()
            for i, tup in enumerate(train_loader, 1):
                _, images = tup['case_name'], tup['image']
                images = images.type('torch.FloatTensor').to(self.device)

                z = self.E(images)
                generated = self.G(z)
                eg_loss = l1_loss(generated, images)
                losses['eg'].append(eg_loss.item())

                z_prior = two_sided(torch.rand_like(z, device=self.device))
                d_z_prior = self.Dz(z_prior.detach())
                d_z = self.Dz(z.detach())

                valid = Variable(torch.FloatTensor(z.shape[0], 1).to(self.device).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(z.shape[0], 1).to(self.device).fill_(0.0), requires_grad=False)

                dz_loss_prior = bce_with_logits_loss(d_z_prior, valid)
                dz_loss = bce_with_logits_loss(d_z, fake)
                dz_loss_tot = (dz_loss + dz_loss_prior)
                losses['dz'].append(dz_loss_tot.item())

                ez_loss = 0.0001 * bce_with_logits_loss(d_z, valid)
                ez_loss.to(self.device)
                losses['ez'].append(ez_loss.item())

                d_i_input = self.Dimg(images.detach())
                d_i_output = self.Dimg(generated.detach())

                di_input_loss = bce_with_logits_loss(d_i_input, valid)
                di_output_loss = bce_with_logits_loss(d_i_output, fake)
                di_loss_tot = (di_input_loss + di_output_loss)
                losses['di'].append(di_loss_tot.item())

                dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, valid)
                losses['dg'].append(dg_loss.item())

                # Start back propagation
                self.eg_optimizer.zero_grad()
                loss = eg_loss + ez_loss + dg_loss
                loss.backward(retain_graph=True)
                self.eg_optimizer.step()

                self.dz_optimizer.zero_grad()
                dz_loss_tot.backward()
                self.dz_optimizer.step()

                self.di_optimizer.zero_grad()
                di_loss_tot.backward()
                self.di_optimizer.step()

                if i % 10 == 0:
                    print('batch {}'.format(i))
                    sys.stdout.flush()

                logging.info('iteration %d : loss : %f, Dz_loss: %f, Di_loss: %f' % (i, loss.item(), dz_loss_tot.item(), di_loss_tot.item()))

                # validation
                with torch.no_grad():
                    if loss.item() < consts.SAVE_MIN_LOSS:
                        cp_path = self.save(where_to_save, to_save_models=True)
                        consts.SAVE_MIN_LOSS = loss.item()

                    if i % 10 == 0:
                        self.eval()
                        for ii, tup in enumerate(valid_loader, 1):
                            _, images = tup['case_name'], tup['image']
                            images = images.type('torch.FloatTensor').to(self.device)
                            z = self.E(images)
                            generated = self.G(z)
                            loss = l1_loss(images, generated)
                            save_image_normalized(images=images, generated=generated, save_name='result_visualization.png')
                            break

    def calculate_latent_features(self, args):
        # sample_name_list = [file for file in os.listdir(os.path.join(os.getcwd(), args.input_data_path))
        #                     if os.path.isfile(os.path.join(os.getcwd(), args.input_data_path, file))]
        sample_name_list = np.load('./test_data/physical_fts.npy', allow_pickle=True).item()['file_name']
        dataset = Embryoid_dataset(base_dir=args.input_data_path, sample_name_list=sample_name_list,
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]))

        self.eval()
        fts = np.zeros((len(sample_name_list), consts.NUM_Z_CHANNELS), float)
        for ii in range(len(sample_name_list)):
            images = dataset[ii]['image'].unsqueeze(0).type('torch.FloatTensor').to(self.device)
            fts[ii,:] = self.E(images).cpu().detach().numpy()


            # # visualization
            # generated = self.G(self.E(images))
            # pdb.set_trace()
            # plt.imshow(dataset[ii]['image'][0,:,:].cpu().detach())
            # plt.axis('off')
            # plt.show()
            # plt.imshow(generated[0,0,:,:].cpu().detach())
            # plt.axis('off')
            # plt.show()
            # pdb.set_trace()
        return fts

    def generate_img(self, lat_fts, ti):
        lat_fts = torch.tensor(lat_fts).type('torch.FloatTensor').to(self.device)
        self.eval()
        generated = self.G(lat_fts)[0, 0,:,:].cpu().detach().numpy()

        fig, ax1 = plt.subplots(1, 1)
        plt.imshow(generated)
        plt.text(5,15,'Time:{}h'.format(ti*0.01+21), c='w')
        plt.axis('off')
        height_, width_ = generated.shape
        fig.set_size_inches(width_ / 100.0, height_ / 100.0)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        if not os.path.exists('./evolution'):
            os.makedirs('./evolution')
        plt.savefig('./evolution/{}.jpg'.format(ti*0.01+21))
        plt.close()



    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.
        """

        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn('eval')

    def train(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn('train')

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            print_timestamp("Saved {} to {}".format(', '.join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        pdb.set_trace()
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.

        :return:
        """
        loaded = []
        # pdb.set_trace()
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname, map_location=torch.device('cuda'))())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))


