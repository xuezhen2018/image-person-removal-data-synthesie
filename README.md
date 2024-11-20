# Image-person-removal-data-synthesie
This repository contains the evaluation code and dataset for our paper "Evaluation of Image Person Removal Tasks: When Data Synthesis Encounters 3D Reconstruction".

# Synthesized Dataset
We are excited to share a collection of four datasets focused on 3D character virtual synthesis, each curated with a distinct approach and dataset size. These datasets are carefully split into training and testing sets in a 7:3 ratio to support efficient machine learning model training and validation.Please use BaiduDisk to download the following dataset.

**VS_3d_500: Manually Collected 3D Reconstructed Characters**\
The VS_3d_500 dataset contains 500 images of 3D reconstructed characters that have been manually collected.**Link:** [VS_3d_500](https://pan.baidu.com/s/1x-aiuX5OS-nKZXCECqNRVw?pwd=je92)\
**VS_3d_7500: Fully Automated Multi-Angle Lighting 3D Reconstruction Characters**\
Comprising 7500 images, the VS_3d_7500 dataset is fully automated and captures 3D reconstructed characters under diverse lighting conditions.**Link:**[VS_3d_7500](https://pan.baidu.com/s/1bkHJarOqNm1LlZfZ7TKUZw?pwd=tf4k)\
**VS_3dBL_500: Optimal Lighting Virtual Synthesis**\
The VS_3dBL_500 dataset, derived from the VS_3d_7500 dataset through critical search, consists of 500 images that have been harmonized for optimal lighting conditions. **Link:** Download [VS_3dBL_500](https://pan.baidu.com/s/1zop6WjKXx37bYAVD-oZnZA?pwd=j1cj)\
**VS_3dH_500: Harmonized Lighting Virtual Synthesis**\
Built upon the VS_3d_500 dataset, the VS_3dH_500 dataset includes 500 images that have undergone a harmonization process through an image coordination network to achieve consistent lighting. **Link:** Download [VS_3dH_500](https://pan.baidu.com/s/1-cLuTHrH_TmHbvSTgSVAsQ?pwd=dwv7)

# Evaluation Code
We have conducted evaluation experiments on the virtual synthesis datasets using four adjusted network models that are compliant with the person removal framework: CTSDG_PR, AOT-GAN_PR, SLBR_PR, and MEDEF_PR. 

# Acknowledgement
This repo is based on [CTSDG](https://github.com/xiefan-guo/ctsdg), [AOT-GAN](https://github.com/researchmm/AOT-GAN-for-Inpainting), [SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal), [MEDEF](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE) and [PensonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset). Many thanks to the excellent repo.
