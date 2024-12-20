![table2](https://github.com/user-attachments/assets/a3980eda-2bff-421c-be28-e508609fd2d7)# Image-person-removal-data-synthesie
This repository contains the code and dataset for our paper "Enhancing Image Person Removal Tasks through 3D Reconstruction and Data Synthesis".
# Prerequisites
python >= 3.6\
PyTorch >= 1.10
# Getting Started
## 1.Installation
- Clone this repo:
```
git clone https://github.com/xuezhen2018/image-person-removal-data-synthesie.git
cd image-person-removal-data-synthesie
```
- Install python requirements:
```
pip install -r requirements.txt
```
## 2.Datasets Preparation
We suggest placing all training data in a folder under the root directory for easy management. Therefore, first create a folder named 'PR_Dataset' and then download the dataset provided below.
```
mkdir PR_Dataset
```
We are excited to share a collection of four datasets focused on 3D character virtual synthesis, each curated with a distinct approach and dataset size. These datasets are carefully split into training and testing sets in a 7:3 ratio to support efficient machine learning model training and validation.

**VS_3d_500: Manually Collected 3D Reconstructed Characters**\
The VS_3d_500 dataset contains 500 images of 3D reconstructed characters that have been manually collected. **Link:** [VS_3d_500(Google Drive)](https://drive.google.com/file/d/1-8bIEhEhmI3fFVF9y6nKmatonB4nSAao/view?usp=drive_link)\
**VS_3d_7500: Fully Automated Multi-Angle Lighting 3D Reconstruction Characters**\
Comprising 7500 images, the VS_3d_7500 dataset is fully automated and captures 3D reconstructed characters under diverse lighting conditions. **Link:**[VS_3d_7500(Baidu Netdisk)](https://pan.baidu.com/s/1bkHJarOqNm1LlZfZ7TKUZw?pwd=tf4k)\
**VS_3dBL_500: Optimal Lighting Virtual Synthesis**\
The VS_3dBL_500 dataset, derived from the VS_3d_7500 dataset through critical search, consists of 500 images that have been harmonized for optimal lighting conditions.  **Link:**[VS_3dBL_500(Google Drive)](https://drive.google.com/file/d/1wYknfAl9Yj8sgEWNsRcsStx_4exCfXyx/view?usp=drive_link)\
**VS_3dH_500: Harmonized Lighting Virtual Synthesis**\
Built upon the VS_3d_500 dataset, the VS_3dH_500 dataset includes 500 images that have undergone a harmonization process through an image coordination network to achieve consistent lighting.  **Link:** [VS_3dH_500(Google Drive)](https://drive.google.com/file/d/1h95zjd7u6tUZp_3du9DWb9kb1pbbAh7m/view?usp=sharing)\

### Dataset Structure
The dataset structure for the four datasets is as follows

  ```
DatasetName/
|
├── GT/          # Training set ground truth
│   ├── 1.png
│   ├── 2.png
│   ├──...
|   └── 350.png
|
├── GT_val/      # validation set ground truth
│   ├── 351.png
│   ├── 352.png
│   └── ...
│   
└── input/       # training set input images
│   ├── 1.png
│   ├── 2.png
│   ├──...
|   └── 350.png
|
├── input_val/   #validation set input images
│   ├── 351.png
│   ├── 352.png
│   └── ...
|
├── mask/        #training set masks
│   ├── 1.png
│   ├── 2.png
│   ├──...
|   └── 350.png
|
└── mask_val/   #validation set masks
│   ├── 351.png
│   ├── 352.png
│   └── ...
  ```


## 3.Training
We have conducted evaluation experiments on the virtual synthesis datasets using four adjusted network models that are compliant with the person removal framework: CTSDG_PR, AOT-GAN_PR, SLBR_PR, and MEDEF_PR. \
Before training, please set the relative paths for the training set and the validation set. The parameters that need to be modified include `--input_root`, `--GT_root`, `--mask_root`, `--val_input_root`, `--val_GT_root`, and `--val_mask_root`. At the same time, you also need to set the path for saving the model, and the parameter that needs to be modified is `--save_dir`.
- For CTSDG_PR
 The file that needs to be modified for parameters is located at `CTSDG_PR/options/train_options.py`.
  ```
  cd CTSDG_PR\
  python train.py \
  ```
- For AOT-GAN_PR
 The file that needs to be modified for parameters is located at `AOT-GAN_PR/utils/option.py`. 
  ```
  cd AOT-GAN_PR\
  python train.py \
  ```
- For SLBR_PR
The file that needs to be modified for parameters is located at `SLBR_PR/train_options.py`.
  ```
  cd SLBR_PR\
  python train.py \
  ```
- For MEDFE_PR
  The file that needs to be modified for parameters is located at `MEDFE_PR/options/base_options.py`.Due to the characteristics of the model, structured images are required as input first. Please process the images according to the Dataset Preparation in the [MEDFE](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE) profile, so the file parameters also include modifying `--st_root` and `--val_st_root`
  ```
  cd MEDFE_PR\
  python train.py \
  ```
## 4.Testing
Before training, please set the relative paths for the training set and the validation set. The parameters that need to be modified include `--val_input_root`, `--val_GT_root`, and `--val_mask_root`. You also need to set the path for loading the model, and the parameter that needs to be modified is `--checkpoint`.
- For CTSDG_PR
 The file that needs to be modified for parameters is located at `CTSDG_PR/options/test_options.py`.
  ```
  cd CTSDG_PR\
  python test.py \
  ```
- For AOT-GAN_PR
 The file that needs to be modified for parameters is located at `AOT-GAN_PR/utils/option.py`. 
  ```
  cd AOT-GAN_PR\
  python test.py \
  ```
- For SLBR_PR
The file that needs to be modified for parameters is located at `SLBR_PR/test_options.py`.
  ```
  cd SLBR_PR\
  python test.py \
  ```
- For MEDFE_PR
  The file that needs to be modified for parameters is located at `MEDFE_PR/options/base_options.py`.
  ```
  cd MEDFE_PR\
  python test.py \
  ```
## 5.Result
In this part, we performed cross evaluation, where several object removal models are selected and trained on different training samples. Then, these models are tested on various test samples. Experimental results are summarized in Table 1.\
`RS, VSv, and VSr(VS_3d_500)` represent datasets created using real person materials, virtual character models, and 3D reconstructed character models under fixed lighting conditions, respectively. The first two are derived from datasets that were produced in our previous work.`Place2` is commonly used as a dataset for training in image inpainting projects. Among these four baseline models, all except for SLBR have been trained with models derived from Place2. The table provided allows for a comparison that demonstrates the method and dataset proposed in this paper significantly enhance the person removal effects compared to existing image inpainting methods. The relatively poor performance of CTSDG may be attributed to a lack of sufficient data, as `CTSDG` is a GAN (Generative Adversarial Network) model that requires a substantial amount of data for training, and our dataset is much smaller in scale compared to Place2. Experiments validating the stability of the network models are presented in our paper.\
**Table 1** Cross evaluation of image person removal models on various datasets.
![table1](https://github.com/user-attachments/assets/7d49ca39-fd94-4f9d-a8e8-88b0e560751c)
Here, we further analyze the factors that affect the quality of the 3D reconstruction dataset. As shown in Table 2, we use the aforementioned four models as the training sets and VS_3d_500 as the validation set. We have examined three conditions, including image harmonization, lighting optimization, and the size of the dataset. We found that when the amount of data is small, the effect of image harmonization is not significant and may even be counterproductive. We speculate that under fixed lighting conditions, the use of image harmonization may lead to a more severe loss of data modalities. Secondly, lighting optimization continues to be effective, which is better than using fixed lighting conditions, with or without image harmonization, at the same level of data. Finally, the number of images in the synthesized data significantly impacts the training effect, and an increase in the volume of data means a significant enhancement in the training effect.
**Table 2** Ablation study on the 3D reconstruction dataset. 
![table2](https://github.com/user-attachments/assets/04e3952e-7446-4b09-bb5a-02173c9b31d0)


# Data Set Creation Method
We refer to this synthesis method as PRDS (Person-Removal Data Synthesis), and a detailed introduction to this method can be found in [PRDS](https://github.com/xuezhen2018/image-person-removal-data-synthesie/tree/main/PRDS).

# Acknowledgement
This repo is based on [CTSDG](https://github.com/xiefan-guo/ctsdg), [AOT-GAN](https://github.com/researchmm/AOT-GAN-for-Inpainting), [SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal), [MEDFE](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE) and [PensonX](https://github.com/sxzrt/Instructions-of-the-PersonX-dataset). Many thanks to the excellent repo.
