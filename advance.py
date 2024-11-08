import numpy as np
from numpy.linalg import eig,inv
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import os
import pickle
from torchvision import datasets
from imageio import  imwrite
##最终的实验数据
def diagonalize(A):
##求矩阵的对角化
    assert A.ndim==2  ##是一个二维矩阵
    assert A.shape[0]==A.shape[1]##为方阵
    eigenvalues,eigenvectors=eig(A)
    return eigenvalues,eigenvectors
##10-11,
# 加载图像
import os
from PIL import Image


def get_first_image_name(folder_path):
    # 获取文件夹中所有图片文件
    images = [img for img in os.listdir(folder_path) if
              os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
    # 如果文件夹为空或者没有图片，返回None
    if not images:
        return None
    # 按字母顺序排序图片文件名
    images.sort()
    # 返回第一个图片的完整文件名
    return os.path.join(folder_path, images[0])

def get_last_image_name(folder_path):
    # 获取文件夹中所有图片文件
    images = [img for img in os.listdir(folder_path) if
              os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
    # 如果文件夹为空或者没有图片，返回None
    if not images:
        return None
    # 按字母顺序排序图片文件名
    images.sort()
    # 返回第一个图片的完整文件名
    return os.path.join(folder_path, images[-1])


def get_all_subdirs(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

import numpy as np
from numpy.linalg import eig,inv
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import os
import pickle
from torchvision import datasets
from imageio import  imwrite

def diagonalize(A):
##求矩阵的对角化
    assert A.ndim==2  ##是一个二维矩阵
    assert A.shape[0]==A.shape[1]##为方阵
    eigenvalues,eigenvectors=eig(A)
    return eigenvalues,eigenvectors
##10-11,
# 加载图像
import os
from PIL import Image


def get_first_image_name(folder_path):
    # 获取文件夹中所有图片文件
    images = [img for img in os.listdir(folder_path) if
              os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
    # 如果文件夹为空或者没有图片，返回None
    if not images:
        return None
    # 按字母顺序排序图片文件名
    images.sort()
    # 返回第一个图片的完整文件名
    return os.path.join(folder_path, images[0])

def get_last_image_name(folder_path):
    # 获取文件夹中所有图片文件
    images = [img for img in os.listdir(folder_path) if
              os.path.splitext(img)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
    # 如果文件夹为空或者没有图片，返回None
    if not images:
        return None
    # 按字母顺序排序图片文件名
    images.sort()
    # 返回第一个图片的完整文件名
    return os.path.join(folder_path, images[-1])


def get_all_subdirs(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

# 使用示例
directory = './'
subdirs = get_all_subdirs(directory)
print(subdirs)
# 使用示例
for i in range(len(subdirs)):
    folder_path = subdirs[i]  # 替换为你的文件夹路径
    first_image_name = get_first_image_name(folder_path)


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


def sine_distance_complex(vec_a, vec_b):
    """计算两个复数向量之间的余弦距离"""
    # 计算复数的点积
    dot_product = np.vdot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # 若任一向量的模为0，抛出异常
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero vector found; cosine distance is undefined.")

    # 计算余弦相似度
    cosine_similarity = np.abs(dot_product) / (norm_a * norm_b)

    return 1-abs(cosine_similarity)


import cmath


def complex_vector_sin_value(vec1, vec2):
    # 假设vec1和vec2是同样长度的复数列表
    if len(vec1) != len(vec2):
        raise ValueError("Complex vector lengths must be equal.")

    sin_values = [cmath.sin(cmath.phase(v1) - cmath.phase(v2)) for v1, v2 in zip(vec1, vec2)]
    return sin_values


"""# 示例使用
vec1 = [complex(1, 1), complex(2, 2), complex(3, 3)]
vec2 = [complex(-1, 1), complex(0, 2), complex(1, 3)]

sin_values = complex_vector_sin_value(vec1, vec2)
print(sin_values)"""


def matrix_sine_distances_complex(matrix_a, matrix_b):
    """计算两个复数矩阵对应行之间的正弦距离"""
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError("The number of columns in both matrices must be the same.")

    distances = []
    for i in range(matrix_a.shape[0]):
        distance = sine_distance_complex(matrix_a[i], matrix_b[i])
        distances.append(distance)

    return np.array(distances)

# 使用示例
directory = './'
subdirs = get_all_subdirs(directory)
subdirs.remove('.idea')
print(subdirs)
first_names=[]
last_names=[]
# 使用示例
for i in range(len(subdirs)):
    folder_path = subdirs[i]  # 替换为你的文件夹路径
    first_image_name = get_first_image_name(folder_path)
    first_names.append(first_image_name)
    last_image_name=get_last_image_name(folder_path)
    last_names.append(last_image_name)
print(first_names)
print(len(first_names))
print(len(last_names))
dir_name=[]
results=[]
for i in range(len(first_names)):
    for j in range(len(last_names)):
        if(i!=j):
            gray_img_ = get_image(last_names[i])
            eigenvalues, eigenvectors = diagonalize(gray_img_)
            gray_img = get_image(first_names[j])
            eigenvalues4, eigenvectors4 = diagonalize(gray_img)
            distances1 = matrix_sine_distances_complex(eigenvectors,eigenvectors4)
            print(i)
            # 找到最小距离的行索引
            closest_index = np.argmin(distances1)
            # 输出最相似的行
            closest_row = eigenvectors[closest_index]
            min_distance = np.min(distances1)
            print(min_distance)
            if(min_distance<0.80):
                results.append(min_distance)

print(len(results))
