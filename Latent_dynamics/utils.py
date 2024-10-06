import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import zoom
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch


import pdb


# funcs for Autoencoder
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'].astype(np.float32), sample['mask']

        # process 4 channels together
        n, x, y = np.shape(image)
        if x != self.output_size[0] or y != self.output_size[1]:
            image_resize = np.zeros((n, self.output_size[0], self.output_size[1]), float)
            for ni in range(n):
                image[ni,:,:][np.where(mask==0)] = 0
                image[ni,:,:] = image[ni,:,:]/np.max(image[ni,:,:])
                image_resize[ni,:,:] = zoom(image[ni,:,:], (self.output_size[0]/x, self.output_size[1]/y), order=0)
            mask = zoom(mask, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        image = torch.from_numpy(image_resize.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        sample = {'image': image[2:3,:,:], 'mask': mask.long()}    # RGBA 4-channel img, only consider B-channel (DAPI)

        return sample


class Embryoid_dataset(Dataset):
    def __init__(self, base_dir, sample_name_list, transform=None):
        self.transform = transform
        self.sample_list = sample_name_list
        self.base_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = '{}'.format(self.sample_list[idx])
        data_path = os.path.join(self.base_dir, slice_name)
        data = np.load(data_path, allow_pickle=True)
        image, mask = data['image'], data['mask']
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx]

        return sample


def save_image_normalized(images, generated, save_name):
    fig, axarr = plt.subplots(4, 5, figsize=(80, 30))
    count = 0
    for row_start in [0,2]:
        for col in range(5):
            axarr[row_start, col].imshow(images.detach().cpu()[count, 0, :, :])
            axarr[row_start+1, col].imshow(generated.detach().cpu()[count, 0, :, :])
            count += 1
    plt.axis('off')
    plt.savefig(save_name)
    plt.close(fig)


def two_sided(x):
    return 2 * (x - 0.5)


fmt_t = "%H_%M"


def print_timestamp(s):
    print("[{}] {}".format(datetime.datetime.now().strftime(fmt_t.replace('_', ':')), s))


def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )



# funcs for GMM
def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = np.zeros(mat_a.shape)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :] #.squeeze()
        # import pdb
        # pdb.set_trace()
        res[:, i, :, :] = np.matmul(mat_a_i, mat_b_i)[:, np.newaxis, :]

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1

    return np.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), axis=2, keepdims=True)