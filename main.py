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
image_path2='img0034/00001.jpg'
gray_img2=get_image(image_path2)
eigenvalues2,eigenvectors2=diagonalize(gray_img2)
print(eigenvectors2)
# 判断两个二维数组是否相同:不同的，但是偏移量一样
are_equal = np.array_equal(eigenvectors2, eigenvectors)

print("两个二维数组完全相同:", are_equal)
# 计算欧氏距离
distances = np.linalg.norm(eigenvectors - eigenvectors2, axis=0)

# 找到最小距离的行索引
closest_index = np.argmin(distances)
print("closest_index_set:",closest_index)
print("eigshape:",eigenvectors.shape)
# 输出最相似的行
closest_row = eigenvectors[closest_index]
print(f"最相似的行: {closest_row}")
min_distance = np.min(distances)
print(min_distance)
print(len(closest_row))


