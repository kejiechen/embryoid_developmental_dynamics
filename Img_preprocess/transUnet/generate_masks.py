import random
import numpy as np
import torch
from transUnet.utils import test_single_volume
from transUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from transUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from scipy.ndimage.interpolation import zoom


def resize_img(image):
    x, y, n = image.shape
    if x != 288 or y != 288:
        image_resize = np.zeros((1, n, 288, 288), dtype="uint8")
        for ni in range(n):
            image_resize[0, ni, :, :] = zoom(image[:, :, ni], (288/x, 288/y), order=3)
    return torch.from_numpy(image_resize.astype(np.float32)).cuda()


def inference(model, emb_imgs):
    model.eval()
    pred_masks, images = [], []
    for img_i in range(np.shape(emb_imgs)[0]):
        image = resize_img(emb_imgs[img_i])
        pred_mask = test_single_volume(image, model, classes=3, patch_size=[288, 288])
        pred_masks.append(pred_mask)
        images.append(image.cpu().detach().numpy().astype(int)[0,:,:,:])
    return np.asarray(pred_masks, dtype=np.float32), images


def gen_masks(emb_imgs):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 3
    config_vit.n_skip = 3
    config_vit.patches.size = (16, 16)
    config_vit.patches.grid = (int(288/16), int(288/16))
    net = ViT_seg(config_vit, img_size=288, num_classes=config_vit.n_classes).cuda()

    pred_masks_mean = np.zeros((np.shape(emb_imgs)[0], 288, 288), float)
    for ii, model_i in enumerate(['model1_epoch_63', 'model2_epoch_48', 'model3_epoch_79', 'model4_epoch_55', 'model5_epoch_213']):
        print(model_i)
        net.load_state_dict(torch.load('./transUnet/model/{}.pth'.format(model_i)))
        pred_masks, images_ = inference(net, emb_imgs)
        pred_masks_mean += pred_masks/5.0
    pred_masks_mean[pred_masks_mean>1.5], pred_masks_mean[pred_masks_mean<0.5] = 2, 0
    pred_masks_mean[np.where((pred_masks_mean>0.5)&(pred_masks_mean<1.5))] = 1

    return pred_masks_mean