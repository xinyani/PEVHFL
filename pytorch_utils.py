#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anirban Das
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
import random
import copy
import os, math
from tqdm import tqdm
from PIL import Image
from print_metrics import print_metrics_binary
from sklearn.utils import shuffle

def normalize(x, means=None, stds=None):
    num_dims = x.shape[1]
    if means is None and stds is None:
        means = []
        stds = []
        for dim in range(num_dims):
            m = x[:, dim, :, :].mean()
            st = x[:, dim, :, :].std()
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
            means.append(m.item())
            stds.append(st.item())
        return x , means, stds
    else:
        for dim in range(num_dims):
            m = means[dim]
            st = stds[dim]
            x[:, dim, :, :] = (x[:, dim, :, :] - m)/st
        return x , None, None
    
class CustomTensorDataset(Dataset):
    """
    TensorDataset with support of transforms. 支持转换的TensorDataset。
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        # to sent indices as well : https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/12
        return x, y, index
    def __len__(self):
        return self.tensors[0].size(0)

class MultiViewDataSet(Dataset):
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    def __init__(self, root, data_type, transform=None, target_transform=None, perform_transform=False, datapoints=0, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.x = []
        self.y = []
        self.root = root
        self.x, self.y = shuffle(self.x,self.y, random_state=seed)
        self.classes, self.class_to_idx = self.find_classes(root)
        self.transform = transform
        self.target_transform = target_transform
        self.perform_transform = perform_transform
        self.datapoints = datapoints
        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            for item in os.listdir(root + '/' + label + '/' + data_type):
                views = []
                for view in os.listdir(root + '/' + label + '/' + data_type + '/' + item):
                    views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)
                self.x.append(views)
                self.y.append(self.class_to_idx[label])
        if datapoints>0:
            self.x = self.x[:self.datapoints]
            self.y = self.y[:self.datapoints]
        if perform_transform:
            # perform the transform upfront instead of waiting for later 提前执行转换，而不是等待稍后
            self.x = self.transformDataset(self.x, self.transform)
    # Override to give PyTorch access to any image on the dataset 覆盖以允许PyTorch访问数据集上的任何图像
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []
        if not self.perform_transform:
            for view in orginal_views:
                im = Image.open(view)
                im = im.convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                views.append(im)
            return views, self.y[index], index
        else:
            # if the transform has already been performed 如果变换已经执行
            return orginal_views, self.y[index], index
    # Override to give PyTorch size of dataset 重写以指定PyTorch数据集的大小
    def __len__(self):
        return len(self.x)
    def transformDataset(self, data, transform):
        print("Transforming Dataset using ", transform)
        res = []
        for sample in tqdm(data):
            images = []
            for view in sample:
                im = Image.open(view)
                im = im.convert('RGB')
                im = transform(im)
                images.append(im)
            res.append(images)
        return res
    
class CifarNet(nn.Module):
    def __init__(self, ensemble=False):
        super(CifarNet, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,3))
        self.fc1 = nn.Linear(128*5*2, 512)
        self.fc2 = nn.Linear(512, 256)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*5*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

class CifarNetCombined(nn.Module):
    def __init__(self, ensemble=False, nb_classes=10, bias=False):
        super(CifarNetCombined, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,3))
        self.fc1 = nn.Linear(128*5*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, nb_classes, bias=bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 128*5*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class CifarNetSimpleSmaller(nn.Module):
    def __init__(self, nb_classes=10, bias=False):
        # similar to https://www.tensorflow.org/tutorials/images/cnn
        super(CifarNetSimpleSmaller, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), padding=2)
        self.do1 = nn.Dropout(p=0.5)
        self.do2 = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=(3,3), padding=2)
        self.conv3 = nn.Conv2d(64, 64,kernel_size=(3,3))
        self.fc1 = nn.Linear(64 * 7 * 3, 64)
        self.do3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, nb_classes, bias=bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.do1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.do2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.do3(x)
        x = self.fc2(x)
        return x

class CifarNet2(nn.Module):
    def __init__(self):
        super(CifarNet2, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class MNIST_NET(nn.Module):
    def __init__(self, ensemble=False):
        super(MNIST_NET, self).__init__()
        self.ensemble = ensemble
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,3))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5,3))
        self.fc1 = nn.Linear(64*4*2, 256)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64*4*2)
        x = self.fc1(x)
        return x 
    
    
class TopLayer(nn.Module):
    def __init__(self, linear_size=512, nb_classes=10, bias = False):
        super(TopLayer, self).__init__()
        self.classifier = nn.Linear(256+256, nb_classes, bias=bias)
        
    def forward(self, x):
        x = self.classifier(F.relu(x))
        return x

def add_model(dst_model, src_model):
    """Add the parameters of two models.添加两个模型的参数。
    Args:
        dst_model (torch.nn.Module): the model to which the src_model will be added.
        src_model (torch.nn.Module): the model to be added to dst_model.
    Returns:
        torch.nn.Module: the resulting model of the addition.
    """
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data + dict_params2[name1].data)
    return dst_model

def scale_model(model, scale):
    """Scale the parameters of a model.缩放模型的参数。
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():   #计算两个模型对应参数的欧氏距离
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)

def federated_avg(models: Dict[Any, torch.nn.Module]):
    nr_models = len(models)
    model_list = list(models.values())
    device = torch.device('cuda' if next(model_list[0].parameters()).is_cuda else 'cpu')
    model = copy.deepcopy(model_list[0])
    model.to(device)
    # set all weights and biases of the model to 0
    model = scale_model(model, 0.0)
    for i in range(nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model




def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k 计算指定k值的k个顶部预测的准确性"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0))
        return res
        


def get_train_or_test_loss_simplified_cifar(network_left,network_right,overall_train_dataloader, overall_test_dataloader, report, cord_div_idx=16):
    network_left.eval()  #在训练模型时会在前面加上：model.train() 在测试模型时在前面使用：model.eval() 如果不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况
    network_right.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss = 0
    train_correct = 0
    test_correct= 0
    test_loss = 0
    # report with actual train set 与实际训练集一起报告
    with torch.no_grad():
        for data, target, indices in overall_train_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            output_left = network_left(data_left)
            output_right = network_right(data_right)
            mean = 0
            std = 1
            noise = torch.randn(output_left.size()) * std + mean
            output_left = output_left + noise.to(device)
            noise = torch.randn(output_left.size()) * std + mean
            output_right = output_right + noise.to(device)
            output_top = output_right + output_left
            # test loss is the average loss of the two clients
            train_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            pred = output_top.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.long().data.view_as(pred)).sum()
        train_loss /= len(overall_train_dataloader)
        report["train_loss"].append(train_loss)
        report["train_accuracy"].append(100. * train_correct / len(overall_train_dataloader.dataset))
        print('\nEntire Training set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, train_correct, len(overall_train_dataloader.dataset),
            100. * train_correct / len(overall_train_dataloader.dataset)))
    # report with actual test set
    with torch.no_grad():
        for data, target, indices in overall_test_dataloader:
            data_left, data_right, target = data[:, :, :, :cord_div_idx].to(device), data[:, :, :, cord_div_idx:].to(device), target.to(device)
            output_left = network_left(data_left)
            output_right = network_right(data_right)

            output_top = output_right + output_left
            # test loss is the average loss of the two clients
            test_loss += F.cross_entropy(output_top, target.long()).item()
            # for accuracy we blindly choose the prediction from pred
            # 输出预测类别
            pred = output_top.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.long().data.view_as(pred)).sum()
        test_loss /= len(overall_test_dataloader)
        report["test_loss"].append(test_loss)
        report["test_accuracy"].append(100. * test_correct / len(overall_test_dataloader.dataset))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, test_correct, len(overall_test_dataloader.dataset),
            100. * test_correct / len(overall_test_dataloader.dataset)))


if __name__ == "__main__":
    one = nn.Conv2d(20,13, 3)
    two =nn.Conv2d(20,13, 3)
    three = nn.Conv2d(20,13, 3)
    bb = federated_avg({1:one, 2:two, 3:three})
    assert torch.isclose(bb.weight.data, (one.weight.data + two.weight.data + three.weight.data)/3.0).sum() == bb.weight.data.numel()
