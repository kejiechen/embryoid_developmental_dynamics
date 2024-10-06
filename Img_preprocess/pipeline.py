import os, re
import pdb

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage import filters, measure, util
from skimage.segmentation import mark_boundaries
from scipy.ndimage import binary_fill_holes
from scipy import ndimage

from funcs.skimage_dtype_version_x import img_as_float
from transUnet.generate_masks import gen_masks
from funcs.emb_prop_funcs import cal_layer_thickness


class preprocess_imgs:
    def __init__(self, root_path):
        self.root_path = root_path
        self.channels = ['w1405', 'w2488', 'w3561', 'w4640']
        self.num_embryoids = 0
        self.process_file_names = []
        self.raw_imgs_4ch = []          # 4-channel images read from the root_path
        self.adjusted_imgs_4ch = []     # 4-channel images cropped, adjusted size, aligned
        self.masks = []                 # masks for tissue and cyst areas

    def get_img_file_names(self):
        return [file_ for file_ in [ii for ii in os.listdir(self.root_path) if
                                    os.path.isfile(os.path.join(self.root_path, ii))] if 'w1405' in file_]

    def get_raw_imgs_4ch(self, file_name):
        name1_reg = re.compile(r'JC\d{6}_t\d{3}_g\d{2}_n\d{1,2}_')
        name2_reg = re.compile(r'_s\d{1,4}.TIF')
        name1 = name1_reg.search(file_name).group(0)
        name2 = name2_reg.search(file_name).group(0)
        for ii, chi in enumerate(self.channels):
            self.raw_imgs_4ch.append(img_as_float(Image.open(os.path.join(self.root_path, '{}{}{}'.format(name1, chi, name2)))))
        self.num_embryoids += 1
        self.process_file_names.append(file_name)

    def adjust_imgs(self):     # crop, move embryoid to the center, adjust img size, alignment
        region_level = 4       # this parameter can be tuned for each group of images

        for emb_i in range(self.num_embryoids):
            regions = np.zeros((512, 512), int)
            raw_imgs_4ch_temp = [self.raw_imgs_4ch[emb_i*4+chi].copy() for chi in range(4)]
            for ii, chi in enumerate(self.channels):
                raw_imgs_4ch_temp[ii][np.where(raw_imgs_4ch_temp[ii] == 0)] = np.mean(raw_imgs_4ch_temp[ii])
                smooth = filters.gaussian(raw_imgs_4ch_temp[ii], sigma=4)
                thresholds = filters.threshold_multiotsu(smooth, classes=4)
                regions += np.digitize(smooth, bins=thresholds)

            mask = regions.copy()
            mask[mask<region_level], mask[mask>=region_level] = 0, 1
            # delete isolated small areas
            mask = measure.label(mask)
            table = measure.regionprops_table(mask, properties=('label', 'area'))
            condition = (table['area'] > 500)  # when area_size<500, delete
            filtered_mask = util.map_array(mask, table['label'], table['label']*condition)
            # fill holes
            filtered_mask = ndimage.binary_fill_holes(filtered_mask, structure=np.ones((10,10))).astype(int)
            # identify area center
            properties = measure.regionprops(filtered_mask)
            ref = properties[0].centroid
            rot_ang = properties[0].orientation

            rot_temp = []
            for ii, chi in enumerate(self.channels):
                # set background color as the min color of the tissue outer edge pixels
                edges = mark_boundaries(filtered_mask, filtered_mask, mode='outer')[:, :, 0]
                ref_value = np.min([np.min(raw_imgs_4ch_temp[ii][edges==1]), np.min(raw_imgs_4ch_temp[ii][filtered_mask>0])*0.95])
                raw_imgs_4ch_temp[ii][filtered_mask==0] = ref_value

                img_chi = np.ones((812, 812), float)*ref_value
                img_chi[150:-150, 150:-150] = raw_imgs_4ch_temp[ii].copy()  # set borders (for crop)

                img_chi = img_chi-np.min(img_chi)  # normalization
                img_chi = img_chi/np.max(img_chi)*255
                rot_temp.append(Image.fromarray(img_chi[int(ref[0]):int(ref[0])+300, int(ref[1]):int(ref[1])+300].astype('uint8')))

            # alignment
            rot_temp = Image.merge("RGBA", [rot_temp[2], rot_temp[1], rot_temp[0], rot_temp[3]])
            self.adjusted_imgs_4ch.append(np.array(rot_temp.rotate(-(rot_ang)/np.pi*180)))

    def generate_mask(self):
        self.masks = gen_masks(self.adjusted_imgs_4ch)

    def get_embryoid_properties(self):
        # a simple visualization
        for i in range(self.num_embryoids):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
            ax[0].imshow(self.raw_imgs_4ch[i*4])   # maker order is: DAPI, GATA3, NANOG, T
            ax[1].imshow(self.adjusted_imgs_4ch[i][:,:,2])    # RGBA, R: NANOG, G: GATA3, B: DAPI, A: T
            ax[2].imshow(self.masks[i,:,:])
            plt.show()

        # some useful embryoid properties
        properties = {'file_name':[], 'time':[], 'group':[], 'tissue_area':[], 'lumen_area':[], 'lumen_eccentricity':[],
                      'lumen_equivalent_diameter':[], 'eccentricity':[], 'equivalent_diameter':[], 'major_axis_length':[],
                      'minor_axis_length':[], 'perimeter':[], 'thickness_ratio':[]}

        for i in range(self.num_embryoids):
            time_reg = re.compile(r't\d{1,3}')
            properties['time'].append(int(time_reg.search(self.process_file_names[i]).group(0).strip('t')))
            group_reg = re.compile(r'g\d{1,3}')
            properties['group'].append(int(group_reg.search(self.process_file_names[i]).group(0).strip('g')))
            properties['file_name'].append(self.process_file_names[i])

            mask_temp = self.masks[i,:,:].copy()
            mask_temp[mask_temp==2] = 0.0
            contours = measure.find_contours(mask_temp, 0.0)
            lumen_img = np.zeros_like(mask_temp)
            if len(contours) > 2:
                temp = contours
                contours = []
                contours.append(temp[0])
                contours.append(temp[1])
                for i in range(2,len(temp)):
                    if len(temp[i])>len(contours[0]):
                        contours[0] = temp[i]
                    elif len(temp[i])>len(contours[1]):
                        contours[1] = temp[i]

            if len(contours) == 1:  # no lumen
                properties['lumen_area'].append(0.0)
                properties['lumen_eccentricity'].append(0.0)
                properties['lumen_equivalent_diameter'].append(0.0)

                sph_contour = sorted(contours, key=lambda x: len(x))[0]
                sph_img = np.zeros_like(mask_temp)
                sph_img[np.floor(sph_contour[:, 0]).astype('int'), np.floor(sph_contour[:, 1]).astype('int')] = 1
                sph_img = binary_fill_holes(sph_img).astype(int)
                sph_img_props = measure.regionprops(np.asarray(sph_img))
                thicknesses = cal_layer_thickness(contours[0], None, sph_img_props[0].centroid)

            elif len(contours) == 2:
                lumen_contour = sorted(contours, key=lambda x: len(x))[0]
                lumen_img[np.floor(lumen_contour[:, 0]).astype('int'), np.floor(lumen_contour[:, 1]).astype('int')] = 1
                lumen_img = binary_fill_holes(lumen_img).astype(int)
                lumen_img_props = measure.regionprops(np.asarray(lumen_img))
                properties['lumen_area'].append(lumen_img_props[0].area)
                properties['lumen_eccentricity'].append(lumen_img_props[0].eccentricity)
                properties['lumen_equivalent_diameter'].append(lumen_img_props[0].equivalent_diameter)

                sph_contour = sorted(contours, key=lambda x: len(x))[1]
                sph_img = np.zeros_like(mask_temp)
                sph_img[np.floor(sph_contour[:, 0]).astype('int'), np.floor(sph_contour[:, 1]).astype('int')] = 1
                sph_img = binary_fill_holes(sph_img).astype(int)
                sph_img_props = measure.regionprops(np.asarray(sph_img))
                thicknesses = cal_layer_thickness(contours[0], contours[1], lumen_img_props[0].centroid)
            properties['tissue_area'].append(sph_img_props[0].area - properties['lumen_area'][-1])
            properties['eccentricity'].append(sph_img_props[0].eccentricity)
            properties['equivalent_diameter'].append(sph_img_props[0].equivalent_diameter)
            properties['major_axis_length'].append(sph_img_props[0].major_axis_length)
            properties['minor_axis_length'].append(sph_img_props[0].minor_axis_length)
            properties['perimeter'].append(sph_img_props[0].perimeter)
            properties['thickness_ratio'].append(np.max(thicknesses) / np.min(thicknesses))

            return properties


