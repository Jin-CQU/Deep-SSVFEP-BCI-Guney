"""
基于Guney等人论文的SSVEP BCI深度学习模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSSVEPBCI(nn.Module):
    """
    基于SSVEP的脑机接口深度神经网络。
    
    架构包括:
    1. 输入层
    2. 四个卷积层
    3. 一个全连接层
    4. Softmax输出层
    """
    
    def __init__(self, input_shape, num_classes, dropout_rates=[0.1, 0.1, 0.95]):
        """
        初始化DeepSSVEPBCI模型。
        
        参数:
            input_shape (tuple): 输入数据的形状 (通道, 样本长度, 子带)
            num_classes (int): 类别/字符数量
            dropout_rates (list): 三个dropout层的dropout率
        """
        super(DeepSSVEPBCI, self).__init__()
        
        channels, sample_length, subbands = input_shape
        self.dropout_rates = dropout_rates
        
        # 用于子带组合的第一卷积层
        # 核大小[1, 1]，1个滤波器，权重初始化为1
        self.conv1 = nn.Conv2d(subbands, 1, kernel_size=(1, 1), bias=False)
        
        # 第二卷积层
        # 核大小[通道, 1]，120个滤波器
        self.conv2 = nn.Conv2d(1, 120, kernel_size=(channels, 1))
        
        # 第一个dropout层
        self.dropout1 = nn.Dropout2d(dropout_rates[0])
        
        # 第三卷积层
        # 核大小[1, 2]，120个滤波器，步长[1, 2]
        self.conv3 = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
        
        # 第二个dropout层
        self.dropout2 = nn.Dropout2d(dropout_rates[1])
        
        # 第四卷积层
        # 核大小[1, 10]，120个滤波器，填充'same'
        self.conv4 = nn.Conv2d(120, 120, kernel_size=(1, 10), padding=(0, 5))
        
        # 第三个dropout层
        self.dropout3 = nn.Dropout2d(dropout_rates[2])
        
        # 全连接层
        # 计算卷积后的大小
        conv_output_size = self._calculate_conv_output_size(input_shape)
        self.fc = nn.Linear(120 * conv_output_size, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
    def _calculate_conv_output_size(self, input_shape):
        """
        计算卷积层后的输出大小。
        
        参数:
            input_shape (tuple): 输入数据的形状
            
        返回:
            int: 卷积后的输出大小
        """
        channels, sample_length, subbands = input_shape
        
        # 在conv2之后: 样本长度保持不变 (核大小 [通道, 1])
        size = sample_length
        
        # 在conv3之后: (大小 - 2) / 2 + 1 = 大小 / 2 (步长为2)
        size = size // 2
        
        # 在conv4之后: 大小保持不变 (使用填充'same')
        
        return size
        
    def _initialize_weights(self):
        """
        初始化模型的权重。
        """
        # 将第一层权重初始化为1
        with torch.no_grad():
            self.conv1.weight.fill_(1.0)
            
        # 使用窄正态分布初始化其他层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m != self.conv1:  # 跳过第一层
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        模型的前向传播。
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 输出张量
        """
        # 排列以匹配PyTorch的期望输入格式 (批次, 通道, 高度, 宽度)
        # 期望输入形状: (批次, 子带, 通道, 样本长度)
        x = x.permute(0, 3, 2, 1)
        
        # 第一卷积层
        x = self.conv1(x)
        
        # 第二卷积层
        x = self.conv2(x)
        
        # 第一个dropout层
        x = self.dropout1(x)
        
        # 第三卷积层
        x = self.conv3(x)
        
        # 第二个dropout层
        x = self.dropout2(x)
        
        # ReLU激活
        x = F.relu(x)
        
        # 第四卷积层
        x = self.conv4(x)
        
        # 第三个dropout层
        x = self.dropout3(x)
        
        # 展平以用于全连接层
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


def create_model(input_shape, num_classes, dropout_rates=[0.1, 0.1, 0.95]):
    """
    创建DeepSSVEPBCI模型的实例。
    
    参数:
        input_shape (tuple): 输入数据的形状 (通道, 样本长度, 子带)
        num_classes (int): 类别/字符数量
        dropout_rates (list): 三个dropout层的dropout率
        
    返回:
        DeepSSVEPBCI: 模型实例
    """
    model = DeepSSVEPBCI(input_shape, num_classes, dropout_rates)
    return model