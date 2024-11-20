# class DataBase:
#     '''Python 3 中的类'''
#
#     def __init__(self, id, address):
#         '''初始化方法'''
#         self.id = id
#         self.address = address
#         self.d = {self.id: 1,
#                   self.address: "192.168.1.1",
#                   }
#
#     def __getitem__(self, key):
#         # return self.__dict__.get(key, "100")
#         return key
#
#
# data = DataBase(1, "192.168.2.11")
# print(data["hi"])
# print(data[data.id])


import numpy as np
import cv2
from datasets.dataset import ImageDataset
from torch.utils import data
from utils.misc import sample_data
from utils.ddp import data_sampler
from  torchvision import utils as vutils
import matplotlib.pyplot as plt

image_dataset = ImageDataset(
    "./images/in_put",
    "./images/mask",
    "./images/GT",
    [256,256],
    "train"
)

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=4,
    sampler=data_sampler(
        image_dataset, shuffle=True
    ),
    drop_last=True, num_workers=0, pin_memory=True
)
image_data_loader = sample_data(image_data_loader)
len_data = image_dataset.__len__()  # 获取数据总值

for i in range(len_data):
    inputs, mask, GT = next(image_data_loader)  # 这里还是tensor格式
    filename = "./images/output/"+str(i)+".jpg"
    vutils.save_image(inputs, filename, normalize=True)

