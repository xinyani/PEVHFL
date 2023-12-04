#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision import models
import copy,math
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, r2_score, roc_auc_score
from sklearn.datasets import load_svmlight_file
import sys
import os
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
from typing import Any
from load_cifar_10 import load_cifar_10_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_utils import CustomTensorDataset, normalize, federated_avg, CifarNetSimpleSmaller, get_train_or_test_loss_simplified_cifar
from sam import SAM, enable_running_stats, disable_running_stats
# from models.resnet2 import *
import warnings
warnings.filterwarnings("ignore")
EPSILON = 0.0000001

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #为CPU中设置种子，生成随机数
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU. 如果您使用的是多GPU。
    torch.backends.cudnn.benchmark = False  #衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
    torch.backends.cudnn.deterministic = True #固定随机数种子

def log_time(file, string=""):
    if string == "" :
        with open(file, 'a') as f:
            f.write(f"Started at :{timer()} \n")   #timer定时器
    else:
        with open(file, 'a') as f:
            f.write(f"Finished {string} :{timer()} \n")

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            print("here")
            iterator = iter(iterable)       
       
def sampleQqminibatches(Q, BATCH_SIZE, GLOBAL_INDICES, with_replacement=True, journal=True):

    if not journal:
        minibatches = []
        if with_replacement:    
            for i in range(Q):
                minibatches.append(random.sample(GLOBAL_INDICES, BATCH_SIZE))
        else:
            copy_GLOBAL_INDICES = copy.deepcopy(GLOBAL_INDICES)
            random.shuffle(copy_GLOBAL_INDICES)
            start = 0
            for i in range(Q):
                minibatches.append(copy_GLOBAL_INDICES[start: (start+1)*BATCH_SIZE])
                start+=1
    else:
        minibatches = []
        sampleonce = random.sample(GLOBAL_INDICES, BATCH_SIZE)
        for i in range(Q):
            minibatches.append(sampleonce)
    return minibatches 
        
class CD(object):   #数据中心
    def __init__(self, alpha: float , X, y , index: int, offset: int, device_list: list, average_network: nn.Module):
        self.alpha: float = alpha
        self.costs = []
        self.X = X
        self.y = y
        self.index = index
        self.device_list = device_list
        self.average_network = average_network
        
class Device(object):
    def __init__(self, network: nn.Module, alpha: float , X, y, device_index: int, dc_index: int, offset: int, indices : list, batch_size, transform=None, momentum=0, sampling_with_replacement=False):
        self.alpha: float = alpha
        self.momentum: float = momentum
        self.indices = indices
        self.batch_size = batch_size
        self.X = pd.DataFrame(X.reshape(X.shape[0], 3*32*16))
        self.y = pd.DataFrame(y)
        self.X.set_index(np.array(self.indices), inplace=True)
        self.y.set_index(np.array(self.indices), inplace=True)
        self.device_index = device_index
        self.dc_index = dc_index
        self.offset = offset
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=alpha, momentum=self.momentum)
        self.lastlayer_Xtheta = torch.zeros((len(X), 256))
    
    def reset_optimizer(self):
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.alpha, momentum=self.momentum)
    
    def getBatchFromIndices(self,indices, Qindex):

        current_batch_index = indices[Qindex]
        intersected_data_points = set(current_batch_index).intersection(set(self.indices))
        return self.X.loc[intersected_data_points, :], self.y.loc[intersected_data_points, :], list(intersected_data_points)

def model_norm(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():   #计算两个模型对应参数的欧氏距离
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)

def parse_args():
    """
    Parse command line arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='TDCD CIFAR')
    parser.add_argument('--data', type=int, nargs='?', default=0, help='dataset to use in training.')
    parser.add_argument('--model', type=int, nargs='?', default=0, help='model to use in training.')
    parser.add_argument('--seed', type=int, nargs='?', default=200, help='Random seed to be used.')
    parser.add_argument('--hubs', type=int, nargs='?', default=2, help='Number of hubs in system (N).')
    parser.add_argument('--clients', type=int, nargs='?', default=10, help='Number of workers per hub (K).')
    parser.add_argument('--gepochs', type=int, nargs='?', default=5000, help='Number of global iterations to train for.')
    parser.add_argument('--Q', type=int, nargs='?', default=5, help='Number of local iterations for client.')
    parser.add_argument('--batchsize', type=int, nargs='?', default=256, help='Batch size to use in Mini-batch in each client in each hub per local iteration.')
    parser.add_argument('--lr', type=float, nargs='?', default=0.01, help='Learning rate of gradient descent.')
    parser.add_argument('--evalafter', type=float, nargs='?', default=10, help='Number of steps after which evaluation must be done.')
    parser.add_argument('--withreplacement', action='store_true', help='If batches are to be picked with sampling with replacement.')
    parser.add_argument('--momentum', type=float, nargs='?', default=0, help='Number of local iterations for client.')
    parser.add_argument('--lambduh', type=float, nargs='?', default=0.01, help='Regularization coefficient.')
    parser.add_argument('--resultfolder', type=str, nargs='?', default="./data/results/journal", help='Results Folder.')
    parser.add_argument('--evaluateateveryiteration', action='store_true', help='If set, then we will evaluate every local round. Else we will evaluate every Q rounds.')
    parser.add_argument('--stepLR', action='store_true', default=False, help='If set, then we will decrease LR in some steps. By default this is false and system uses initial LR.')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    # Parse input arguments 分析输入参数
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed_torch(args.seed)
    # Load the a9a dataset. In this case we will be using a constant intercept feature 加载a9a数据集。在这种情况下，我们将使用恒定截距功能
    cifar_10_dir = "./data/cifar10"#'cifar10'
    X_train, train_filenames, y_train, X_test, test_filenames, y_test, label_names = load_cifar_10_data(cifar_10_dir)
    X_train = torch.FloatTensor(X_train)/255.0 #scale all images by 255
    X_train = X_train.permute(0, 3, 1, 2) # to make it 50000, 3, 32, 32
    X_test = torch.FloatTensor(X_test)/255.0 #scale all images by 255
    X_test = X_test.permute(0, 3, 1, 2)# to make it 10000, 3, 32, 32
    """
    We need to standardize the tensor dataset 我们需要标准化张量数据集
    We normalize by : image = (image - mean) / std
    in this case, for cifar10/mnist, we have 
    https://github.com/kuangliu/pytorch-cifar/issues/19
    https://github.com/kuangliu/pytorch-cifar/issues/16
    https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values
    """
    means = torch.tensor([0.4914, 0.4822, 0.4465])
    stds = torch.tensor([0.247, 0.243, 0.261])
    X_train.sub_(means[None, :, None, None]).div_(stds[ None, :, None, None])
    X_test.sub_(means[None, :, None, None]).div_(stds[ None, :, None, None])
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    X_train_numpy = X_train.numpy()
    y_train_numpy = y_train.numpy()
    # 1.a Now we assume that there are N DCs 现在我们假设有N个DC
    N = args.hubs 
    K = args.clients 
    global_epoch = args.gepochs 
    local_epoch = args.Q 
    local_batch_size = args.batchsize 
    coordinate_per_dc = int(X_train.shape[2]/N)
    extradatapointsinfirstdevice = X_train.shape[2] - coordinate_per_dc*N
    datapoints_per_device = int(X_train.shape[0]/(K))
    alpha = args.lr # 0.1 
    momentum = args.momentum
    lambduh = args.lambduh
    decreasing_step = False
    ######################################################################################
    #--------------------DATA DISTRIBUTION FOR EXPERIMENTS 实验数据分布--------------------#
    ######################################################################################
    # 1.b create N DCs and distribute the coordinates between them 创建N个DC并在它们之间分配坐标
    dc_list = []
    global_weights = np.zeros((X_train.shape[2], 1))
    global_indices = list(range(len(X_train)))
    GLOBAL_INDICES = list(range(len(X_train)))

    coordinate_partitions = []
    coordinate_per_dc = int(X_train.shape[2]/N)
    extradatapointsinfirstdevice = X_train.shape[2] - coordinate_per_dc*N
    i = 0
    while i< X_train.shape[2]:
        if extradatapointsinfirstdevice>0:
            coordinate_partitions.append(list(range(i, i+ coordinate_per_dc + 1)))
            extradatapointsinfirstdevice-=1
            i=i+coordinate_per_dc + 1
        else:
            coordinate_partitions.append(list(range(i, i+ coordinate_per_dc )))
            i=i+coordinate_per_dc
    training_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    over_train_loader = torch.utils.data.DataLoader(CustomTensorDataset(tensors=(X_train, y_train), transform=None), batch_size=1000, shuffle=False)
    over_test_loader = torch.utils.data.DataLoader(CustomTensorDataset(tensors=(X_test, y_test), transform=None), batch_size=1000, shuffle=False)
    for i in range(N):
        coordinate_per_dc = len(coordinate_partitions[i])
        dc_X = X_train_numpy[:, :, :, coordinate_partitions[i]]
        # Create a list of device connected to each DC, and suppose all of them have same number of data points which we distribute from here
        # 创建一个连接到每个DC的设备列表，并假设它们都有相同数量的数据点，我们从这里分发这些数据点
        device_list = []
        network_local = models.resnet18(pretrained=True)   #network_local = CifarNetSimpleSmaller(nb_classes=10)  # network_local = ResNet18()
        for k in range(K):
            device_list.append(Device(alpha=alpha,
                                      momentum=momentum,
                                      X=dc_X[k*datapoints_per_device : (k+1) * datapoints_per_device, :, :, :],
                                      y=y_train_numpy[k*datapoints_per_device : (k+1) * datapoints_per_device],
                                      device_index=k,
                                      dc_index=i,
                                      offset=datapoints_per_device,
                                      indices = list(range(k*datapoints_per_device , (k+1) * datapoints_per_device)),
                                      batch_size = local_batch_size, 
                                      network = copy.deepcopy(network_local),
                                      sampling_with_replacement= args.withreplacement
                                ))
        # Create the Data Center and attach the list of devices to it创建数据中心并将设备列表附加到数据中心
        dc_list.append(CD(alpha=alpha, # need very small alpha
                          X=dc_X,
                          y=y_train,
                          index=i,
                          offset=coordinate_per_dc, 
                          device_list=device_list,
                          average_network = copy.deepcopy(network_local)))
    del X_train, y_train

    report = {
              "train_loss": [],
              "test_loss": [],
              "train_accuracy": [],
              "test_accuracy": [],
              "hyperparameters": args
              }
    START_EPOCH = 0
    PATH = (f"Checkpoint_Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}.pt")
    if os.path.exists(PATH):
        print(
            """
            --------------LOADING FROM CHECKPOINT-----------------
            """
        )
        checkpoint = torch.load(PATH)
        START_EPOCH = int(checkpoint['epoch']) + 1 # start from the next epoch 从下一个迭代开始
        for hub_idx in range(N):                   # 从hub获取模型
            dc_list[hub_idx].average_network.load_state_dict(checkpoint["hub_average_network_state_dict"][hub_idx])
            for device_idx, device in enumerate(dc_list[hub_idx].device_list):
                device.network = copy.deepcopy(dc_list[hub_idx].average_network)
                device.reset_optimizer()
        if not args.stepLR:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        else:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr[{alpha},0.005,0.001]_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl"
        f = open(os.path.join(args.resultfolder, filename), "rb")
        report = pickle.load(f)
    for t in range(START_EPOCH, global_epoch):
        print(f"Epoch {t}/{global_epoch}")

        batch_for_round = {}
        batch_indices_and_exchange_info_for_epoch = {i:{} for i in range(N)}
        mini_batch_indices = sampleQqminibatches(local_epoch, args.batchsize, GLOBAL_INDICES, with_replacement=True)
        for k_idx, k in enumerate(range(N)):
            current_DC = dc_list[k_idx]
            otherhub_index = 1 if k_idx == 0 else 0
            for device_idx, device in enumerate(current_DC.device_list):
                batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index] = []
                for iterations in range(local_epoch):

                    temp_X , temp_y, batch_indices = device.getBatchFromIndices(mini_batch_indices, iterations)
                    # assert that the batch indices in this cand other hub are same by checking if the labels are equal or not 通过检查标签是否相等，断言该集线器和其他集线器中的批索引相同
                    np.testing.assert_array_equal(np.array(dc_list[otherhub_index].device_list[device_idx].y.loc[batch_indices]), np.array(device.y.loc[batch_indices]))

                    device.network.to(training_device)
                    with torch.no_grad():
                        if len(temp_X)==0:
                            batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index].append({"batch_indices": copy.deepcopy(batch_indices), "embedding":torch.zeros(1)})
                            continue
                        temp_X = torch.FloatTensor(np.array(temp_X).reshape(temp_X.shape[0], 3, 32, 16))
                        temp_X = temp_X.to(training_device)
                        output = device.network(temp_X)
                        batch_indices_and_exchange_info_for_epoch[k_idx][device.device_index].append({"batch_indices": copy.deepcopy(batch_indices), "embedding":output})
        """
        As there are only two hubs, step 2 and step 3 are incorporated in step 1 由于只有两个hub，步骤2和步骤3包含在步骤1中
        DO THE ACTUAL TRAINING WITH THE ABOVE SELECTED BATCHES FOR THIS GLOBAL ROUND 使用上述选定的批次为本次全局回合进行实际训练
        """            
        for iteration in range(local_epoch):      
            """
            Implement variable learning rate.
            between 0-8000 time steps it is 0.01
            between 8000-16000 time step it is 0.005
            between 16000-30000 time step it is 0.001
            """
            if args.stepLR:
                if t*local_epoch + iteration >= 8000 and t*local_epoch + iteration<16000:
                    print(f"\n\n LE {t*local_epoch + iteration} LR:0.005 ") 
                    for k_idx, k in enumerate(range(N)):
                        current_DC = dc_list[k_idx]
                        for device_idx, device in enumerate(current_DC.device_list):        
                            for g in device.optimizer.param_groups:
                                g['lr'] = 0.005
                elif t*local_epoch + iteration >= 16000:
                    print(f"\n\n LE {t*local_epoch + iteration} LR:0.001 ")
                    for k_idx, k in enumerate(range(N)):
                        current_DC = dc_list[k_idx]
                        for device_idx, device in enumerate(current_DC.device_list):        
                            for g in device.optimizer.param_groups:
                                g['lr'] = 0.001
            for hub_index, k in enumerate(range(N)):     #############################################################训练
                coordinate_per_dc = len(coordinate_partitions[k])
                current_DC = dc_list[hub_index]
                # now learn parallelly in each connected device in current_DC 现在在current_DC中的每个连接设备中并行学习
                # Isolate the H_-k from other datacenters for the same label space 将H_-k与相同标签空间的其他数据中心隔离
                # Obtained in the last iteration 在上次迭代中获得
                # start of local iterations 局部迭代的开始
                """
                Since we are using the same minibatch for Q iterations for the journal 由于我们使用相同的迷你批次进行期刊的Q迭代
                """
                for device_idx, device in enumerate(current_DC.device_list):
                    # select the batch indices from the Q minibatches picked earlier 从前面选择的Q个小批次中选择批次索引
                    device.network.to(training_device)
                    device.network.train()
                    batch_indices = batch_indices_and_exchange_info_for_epoch[hub_index][device_idx][iteration]["batch_indices"]
                    temp_X , temp_y, _ = device.getBatchFromIndices(mini_batch_indices, iteration)
                    temp_X = torch.FloatTensor(np.array(temp_X).reshape(temp_X.shape[0], 3, 32, 16))
                    temp_y = torch.FloatTensor(np.array(temp_y))[:,0]
                    if len(temp_X) ==0:
                        print(f"Client {device.device_index} of {device.dc_index} does not have any datapoints in {t}:{iteration}. \n Skipping this round of training.")
                        continue
                    temp_X , temp_y = temp_X.to(training_device), temp_y.to(training_device)
                    device.optimizer.zero_grad()
                    output = device.network(temp_X)
                    if hub_index==0:
                        output_top_from_other_hub_client = batch_indices_and_exchange_info_for_epoch[1][device.device_index][iteration]["embedding"]
                    elif hub_index==1:    
                        output_top_from_other_hub_client = batch_indices_and_exchange_info_for_epoch[0][device.device_index][iteration]["embedding"]
                    total_output = output+output_top_from_other_hub_client
                    # loss = F.cross_entropy(total_output, temp_y.long())

                    # #######################  fedprox-blur #######################
                    # fed_prox_reg = 0.0
                    # mu = 1
                    #
                    # norm = model_norm(dc_list[hub_index].average_network.to(training_device), device.network)
                    # S = norm
                    # for name, layer in dc_list[hub_index].average_network.named_parameters():
                    #     fed_prox_reg += ((1 / 2) * torch.norm((layer.data.to(training_device) - device.network.state_dict()[name])) ** 2-S**2)

                    # loss += max(0,fed_prox_reg)
                    # print(fed_prox_reg)
                    # ##################################################
                    #
                    # loss.backward()
                    # device.optimizer.step()

                    # # ######################## fedsam  ###########################
                    base_optimizer = torch.optim.SGD
                    # optimizer = SAM(device.network.parameters(), base_optimizer, rho=1-1/(t+1), adaptive=True, lr=0.01, momentum=0.5, weight_decay=5e-4)
                    optimizer = SAM(device.network.parameters(), base_optimizer, rho=0.01, adaptive=True,
                                    lr=0.01, momentum=0.5, weight_decay=5e-4)

                    criterion = nn.CrossEntropyLoss().to(training_device)
                    enable_running_stats(device.network)

                    loss = criterion(device.network(temp_X), temp_y.long())
                    # loss += max(0, fed_prox_reg)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward step
                    disable_running_stats(device.network)
                    criterion(device.network(temp_X), temp_y.long()).backward()
                    optimizer.second_step(zero_grad=True)
                    # # ###################################################################




            """
            After taking one local step in each device in each data center. We calculate the loss 在每个数据中心的每个设备中执行一个本地步骤之后。我们计算损失
            """
            # GENERATE REPORT EVERY Q steps by averaging 平均每Q步生成报告
            # Now generate report every 10 steps 现在每10步生成一次报告
            if args.evaluateateveryiteration or (t*local_epoch + iteration+1)%args.evalafter==0:    ####################测试
                print(f"calculating every {args.evalafter} rounds", local_epoch, iteration, t*local_epoch + iteration)
                averaged_networks = [None]*N
                for hub_index, k in enumerate(range(N)):
                    coordinate_per_dc = len(coordinate_partitions[k])
                    current_DC = dc_list[hub_index]
                    # Average weights for reporting but do not replace the local weights 用于报告但不替换本地权重的平均权重
                    per_batch_model_list = {}
                    # MPI_Reduce within each Data Center to average the model MPI_在每个数据中心内减少以平均模型
                    for device_idx, device in enumerate(current_DC.device_list):
                        per_batch_model_list[device_idx] = copy.deepcopy(device.network) # as if the device sends the model to the DC 就好像设备将模型发送到DC
                    # MPI_Reduce within each Data Center to average the model MPI_在每个数据中心内减少以平均模型
                    averaged_networks[hub_index] = federated_avg(per_batch_model_list)
                """
                DCS exchange the top layer information between eachother without averaging, but concatenating, 
                This allows us to maintain a Oracle like overall top layer network
                DCS在彼此之间通过连接交换顶层信息而不进行平均，这使我们能够维护类似Oracle的整体顶层网络
                """
                # This is the MPI reduce part between the DCs 这是DC之间的MPI减少部分
                # Get train loss at each local iteration for each global iteration 获得每个全局迭代的每个局部迭代的训练损失
                get_train_or_test_loss_simplified_cifar(averaged_networks[0], averaged_networks[1], over_train_loader, over_test_loader, report, cord_div_idx=16)

        for k_idx, k in enumerate(range(N)):  #加权平均
            current_DC = dc_list[k_idx]
            device_model_list = {}
            device_top_layer_model_list = {}
            for device_idx, device in enumerate(current_DC.device_list):
                device_model_list[device_idx] = copy.deepcopy(device.network) # as if the device sends the model to the DC 就好像设备将模型发送到DC
            # MPI_Reduce within each Data Center to average the model MPI_在每个数据中心内减少以平均模型


            sigma = 0.22
            C = 1
            for i in range(device_idx):
                norm = model_norm(dc_list[k_idx].average_network.to(training_device), device_model_list[i])
                norm_scale = min(1, C/ (norm))
                # print(model_norm, norm_scale)
                for name, layer in device_model_list[i].named_parameters():
                    clipped_difference = norm_scale * (layer.data - dc_list[k_idx].average_network.state_dict()[name])
                    layer.data.copy_(device_model_list[i].state_dict()[name] + clipped_difference)
                    noise = torch.cuda.FloatTensor(layer.data.shape).normal_(0, sigma)  # normal正态分布
                    layer.data.add_(noise.long())


            current_DC.average_network = federated_avg(device_model_list)

            for device_idx, device in enumerate(current_DC.device_list):
                device.network = copy.deepcopy(current_DC.average_network)
                device.reset_optimizer()
        """
        Save Report and checkpoint 保存报告和检查点
        """
        PATH = (f"Checkpoint_Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}.pt")
        torch.save({'epoch': t, 'hub_average_network_state_dict' : [i.average_network.state_dict() for i in dc_list],}, PATH)
        # =============================================================================
        os.makedirs(f"{args.resultfolder}", exist_ok=True)
        if not args.stepLR:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr{alpha}_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        else:
            filename =f"Simplified_Cifar_model_BS{local_batch_size}_N{N}_K{K}_Q{local_epoch}_lr[{alpha},0.005,0.001]_momentum{momentum}_seed{args.seed}_sampling{args.withreplacement}_eval{args.evaluateateveryiteration}_evalafter{args.evalafter}.pkl" 
        f = open(os.path.join(args.resultfolder, filename), "wb")
        pickle.dump(report, f)