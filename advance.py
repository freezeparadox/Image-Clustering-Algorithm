import numpy as np
from numpy.linalg import eig,inv
import torch
import torchvision.transforms as transforms
from PIL import Image
def diagonalize(A):
##求矩阵的对角化
    assert A.ndim==2  ##是一个二维矩阵
    assert A.shape[0]==A.shape[1]##为方阵
    eigenvalues,eigenvectors=eig(A)
    return eigenvalues,eigenvectors

# 加载图像
image_path = '00001.jpg'
def get_image(image_path):
    image = Image.open(image_path)

    # 定义图像转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 调整图像大小以便计算
    ])
    image_tensor = transform(image) # 增加批量维度
    print(image_tensor.shape)

    gray_img = torch.mean(image_tensor, dim=0, keepdim=True).squeeze(0)
    print(gray_img.shape)
    return gray_img
gray_img=get_image(image_path)
eigenvalues,eigenvectors=diagonalize(gray_img)
print(len(eigenvalues))

import os
# 指定你要遍历的文件夹路径
folder_path = './boat'

# 初始化一个空数组用于存储文件名
file_names = []
file_paths=[]
# 遍历文件夹中的文件
for file_name in os.listdir(folder_path):
    # 构造文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 检查是否是文件（不是文件夹）
    if os.path.isfile(file_path):
        # 将文件名添加到数组中
        file_names.append(file_name)
        file_path="./boat/"+file_name
        file_paths.append(file_path)
# 打印文件名数组
print(file_names)

print(file_paths)

result=[]
line=[]
for i in range(100):
    gray_img2 = get_image(file_paths[i])
    eigenvalues2, eigenvectors2 = diagonalize(gray_img2)
    distances = np.linalg.norm(eigenvectors - eigenvectors2, axis=1)

    # 找到最小距离的行索引
    closest_index = np.argmin(distances)
    # 输出最相似的行
    closest_row = eigenvectors[closest_index]
    min_distance = np.min(distances)
    print(min_distance)
    result.append(min_distance)
    print(len(closest_row))
    line.append(len(closest_row))

print(result)
print(line)
print(len(line))
print(len(result))
index=[]
for i in range(100):
    if result[i]<0.65:
        index.append(i)

print(len(index))
print(index)
