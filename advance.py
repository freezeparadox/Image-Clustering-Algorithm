import numpy as np
import numpy as np
from numpy.linalg import eig,inv
import torch
import torchvision.transforms as transforms
from PIL import Image,ImageOps
import os
import pickle
from torchvision import datasets


def complex_cosine_similarity(z, w):
    """
    计算两个复数向量的余弦相似度（绝对值版本）
    参数:
        z, w: 一维numpy数组，元素为复数
    返回:
        相似度 (0到1之间的实数)
    """
    # 步骤1 & 2: 计算埃尔米特内积 (np.vdot 自动对第二个参数取共轭)
    inner_product = np.vdot(z, w)  # 等价于 np.dot(z.conj(), w)

    # 步骤3: 计算向量的2-范数
    norm_z = np.linalg.norm(z)  # 默认使用2-范数，对于复数是正确的
    norm_w = np.linalg.norm(w)

    # 步骤4 & 5: 计算相似度
    similarity = np.abs(inner_product) / (norm_z * norm_w)

    return similarity

def not_simlarity(z,w):
    simlarity = complex_cosine_similarity(z, w)
    return 1-simlarity


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

# 示例
z = np.array([1 + 1j, 2 - 1j])
w = np.array([0 + 1j, 1 + 2j])

sim = complex_cosine_similarity(z, w)
print(f"余弦相似度: {sim:.4f}")  # 输出: 0.9386

# 验证：与手动计算一致
print(f"验证: {np.sqrt(37) / np.sqrt(42):.4f}")  # 输出: 0.9386

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

gray_img = get_image(image_path)
eigenvalues,eigenvectors=diagonalize(gray_img)
print(eigenvectors.shape)
#print(eigenvectors[1])#一行有256个值

def matrix_sine_distances_complex(matrix_a, matrix_b):
    """计算两个复数矩阵对应行之间的正弦距离"""
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError("The number of columns in both matrices must be the same.")

    distances = []
    for i in range(matrix_a.shape[0]):
        distance = not_simlarity(matrix_a[i], matrix_b[i])
        distances.append(distance)

    return np.array(distances)


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
    folder_path = "processed_datasets/"+folder_path
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
    folder_path = "processed_datasets/" + folder_path
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
directory = './processed_datasets'
subdirs = get_all_subdirs(directory)
print(subdirs)
# 使用示例
for i in range(len(subdirs)):
    folder_path = subdirs[i]  # 替换为你的文件夹路径
    print("folder_path",folder_path)
    first_image_name = get_first_image_name(folder_path)
    print("first_image_name",first_image_name)

directory = './processed_datasets'
subdirs = get_all_subdirs(directory)
#subdirs.remove('.idea')
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
print(last_names)
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
            if(min_distance<0.75):
                results.append(min_distance)

print(len(results))
