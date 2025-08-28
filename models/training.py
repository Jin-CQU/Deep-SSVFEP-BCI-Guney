"""
DeepSSVEPBCI模型的训练和评估流水线。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from .deep_ssvep_bci import create_model


def calculate_itr(acc_matrix, M, t):
    """
    计算信息传输率(ITR)。
    
    参数:
        acc_matrix (np.ndarray): 准确率矩阵
        M (int): 字符数量
        t (float): 总信号长度(视觉提示+期望信号长度)
        
    返回:
        np.ndarray: ITR矩阵
    """
    size_mat = acc_matrix.shape
    itr_matrix = np.zeros(size_mat)
    
    for i in range(size_mat[0]):
        for j in range(size_mat[1]):
            p = acc_matrix[i, j]
            if p < 1/M:
                itr_matrix[i, j] = 0
            elif p == 1:
                itr_matrix[i, j] = np.log2(M) * (60/t)
            else:
                itr_matrix[i, j] = (np.log2(M) + p*np.log2(p) + (1-p)*np.log2((1-p)/(M-1))) * (60/t)
                
    return itr_matrix


def train_model_first_stage(model, train_data, train_labels, config, device):
    """
    模型的第一阶段训练(全局训练)。
    
    参数:
        model (nn.Module): 要训练的模型
        train_data (np.ndarray): 训练数据
        train_labels (np.ndarray): 训练标签
        config (dict): 配置字典
        device (torch.device): 用于训练的设备
        
    返回:
        nn.Module: 训练后的模型
    """
    # 将数据转换为PyTorch张量
    train_data = torch.FloatTensor(train_data).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(config['max_epochs']):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印进度
        if (epoch + 1) % 100 == 0:
            print('第一阶段 - 轮次 [{}/{}], 损失: {:.4f}'.format(epoch+1, config["max_epochs"], loss.item()))
    
    return model


def train_model_second_stage(model, train_data, train_labels, config, device):
    """
    模型的第二阶段训练(受试者特定微调)。
    
    参数:
        model (nn.Module): 要训练的模型
        train_data (np.ndarray): 训练数据
        train_labels (np.ndarray): 训练标签
        config (dict): 配置字典
        device (torch.device): 用于训练的设备
        
    返回:
        nn.Module: 训练后的模型
    """
    # 将数据转换为PyTorch张量
    train_data = torch.FloatTensor(train_data).to(device)
    train_labels = torch.LongTensor(train_labels).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # 训练循环
    model.train()
    for epoch in range(1000):  # 第二阶段固定1000轮次
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印进度
        if (epoch + 1) % 100 == 0:
            print('第二阶段 - 轮次 [{}/{}], 损失: {:.4f}'.format(epoch+1, 1000, loss.item()))
    
    return model


def evaluate_model(model, test_data, test_labels, device):
    """
    在测试数据上评估模型。
    
    参数:
        model (nn.Module): 要评估的模型
        test_data (np.ndarray): 测试数据
        test_labels (np.ndarray): 测试标签
        device (torch.device): 用于评估的设备
        
    返回:
        tuple: (准确率, 预测结果)
    """
    # 将数据转换为PyTorch张量
    test_data = torch.FloatTensor(test_data).to(device)
    test_labels = torch.LongTensor(test_labels).to(device)
    
    # 评估
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        
        # 计算准确率
        accuracy = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
        
    return accuracy, predicted.cpu().numpy()