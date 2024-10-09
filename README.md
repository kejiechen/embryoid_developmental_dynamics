# embryoid_developmental_dynamics

This repository contains two branches with materials of the Paper "Deep manifold learning characterizes latent developmental dynamics of the stem cell based embryoid model":
* Pre-processing pipeline of the raw embryoid images (./Img_preprocess)
* Analyzing latent feature propagation and inferring continuos embryoid development (./Latent_dynamics)

### Pre-requisites
* Python 3.6 or later
* preferred library versions: pytorch 2.0.1+cu118, scipy 1.11.3, scanpy 1.10.3, scikit-image 0.22.0

### Datasets
* Example raw images (only one embryoid left in the image, others are covered manually) are provided under ./Img_preprocess/test_imgs
* Segemented images and masks are provided under ./Latent_dynamics/test_data
* Physical features (including: file name, group id, time, tissue area, lumen area, thickness ratio, etc.) of all 3,697 images are provided in ./Latent_dynamics/test_data/physical_fts.npy
* 5-dimensional latent features from the autoencoder model of all 3,697 images (DAPI channel) are provided in ./Latent_dynamics/test_data/w1405_dim5_fts.npy

### Citation
If you decide to use materials from this repository in a published work, please kindly cite us using the following bibliography:

@Article{chen, AUTHOR = {...}, TITLE = {Deep manifold learning characterizes latent developmental dynamics of the stem cell based embryoid model}, JOURNAL = {Science Advances}, VOLUME = {...}, YEAR = {...}, NUMBER = {...}, ARTICLE NUMBER = {...}, URL = {...}, DOI = {...} }

