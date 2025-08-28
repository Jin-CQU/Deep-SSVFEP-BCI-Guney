"""
Deep-SSVFEP-BCI项目的主入口点。
"""
import numpy as np
import torch
from data.preprocessing import create_bandpass_filters, preprocess_data
from models.deep_ssvep_bci import create_model
from models.training import train_model_first_stage, train_model_second_stage, evaluate_model, calculate_itr
from utils.config import DATASET_CONFIGS, CHANNELS_9


def load_data(subject, dataset_type='Bench'):
    """
    为特定受试者加载数据。
    
    参数:
        subject (int): 受试者编号
        dataset_type (str): 数据集类型 ('Bench' 或 'BETA')
        
    返回:
        np.ndarray: 加载的数据
    """
    # 这是一个占位函数。在实际实现中，您将从文件中加载实际数据。
    # 数据格式应为：(通道, 样本, 字符, 数据块)
    print("正在为受试者 {} 从 {} 数据集加载数据...".format(subject, dataset_type))
    
    # 目前，我们将创建具有正确形状的虚拟数据
    config = DATASET_CONFIGS[dataset_type]
    channels = config['total_ch']
    samples = int(config['sampling_rate'] * config['visual_cue'] * 2)  # 虚拟大小
    characters = config['totalcharacter']
    blocks = config['totalblock']
    
    # 创建虚拟数据
    data = np.random.rand(channels, samples, characters, blocks)
    return data


def main():
    """
    运行SSVEP BCI流水线的主函数。
    """
    # 配置
    dataset_type = 'Bench'  # 或 'BETA'
    signal_length = 0.4  # 信号长度（秒）
    use_9_channels = True  # 使用9个通道还是全部64个通道
    
    # 获取数据集配置
    config = DATASET_CONFIGS[dataset_type]
    config['signal_length'] = signal_length
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('使用设备: {}'.format(device))
    
    # 创建带通滤波器
    bp_filters = create_bandpass_filters(config['sampling_rate'])
    
    # 计算样本长度和延迟
    sample_length = int(config['sampling_rate'] * signal_length)
    total_delay = config['visual_latency'] + config['visual_cue']
    delay_sample_point = int(total_delay * config['sampling_rate'])
    sample_interval = slice(delay_sample_point, delay_sample_point + sample_length)
    
    # 选择通道
    channels = CHANNELS_9 if use_9_channels else list(range(config['total_ch']))
    
    # 初始化准确率矩阵
    acc_matrix = np.zeros((config['totalsubject'], config['totalblock']))
    
    # 留一数据块交叉验证
    for block in range(config['totalblock']):
        print("\n正在处理数据块 {}/{}...".format(block+1, config['totalblock']))
        
        # 获取除当前数据块外的所有数据块用于训练
        all_blocks = [i for i in range(config['totalblock']) if i != block]
        
        # 加载和预处理训练数据
        print("正在加载和预处理训练数据...")
        train_data_list = []
        train_labels_list = []
        
        for subject in range(1, config['totalsubject'] + 1):
            # 为受试者加载数据
            raw_data = load_data(subject, dataset_type)
            
            # 预处理数据
            processed_data = preprocess_data(raw_data, channels, sample_interval, bp_filters)
            
            # 收集所有训练数据块的数据
            for train_block in all_blocks:
                train_data_list.append(processed_data[:, :, :, :, train_block])
                # 创建标签（为每个字符重复）
                labels = np.repeat(np.arange(config['totalcharacter']), processed_data.shape[3])
                train_labels_list.append(labels)
        
        # 合并所有训练数据
        train_data = np.concatenate(train_data_list, axis=3)
        train_labels = np.concatenate(train_labels_list)
        
        # 重塑数据以适配模型输入
        # 期望形状: (样本, 子带, 通道, 样本长度)
        train_data = train_data.transpose(3, 2, 0, 1)
        
        # 创建模型
        input_shape = (len(channels), sample_length, len(bp_filters))
        model = create_model(input_shape, config['totalcharacter'])
        model.to(device)
        
        # 第一阶段训练（全局模型）
        print("正在进行第一阶段训练...")
        model = train_model_first_stage(model, train_data, train_labels, config, device)
        
        # 保存全局模型
        torch.save(model.state_dict(), 'global_model_block_{}.pth'.format(block+1))
        
        # 第二阶段训练（受试者特定微调）
        print("正在进行第二阶段训练...")
        all_conf_matrix = np.zeros((config['totalcharacter'], config['totalcharacter']))
        
        for subject in range(1, config['totalsubject'] + 1):
            print("  正在为受试者 {}/{} 进行微调...".format(subject, config['totalsubject']))
            
            # 加载和预处理受试者特定训练数据
            raw_data = load_data(subject, dataset_type)
            processed_data = preprocess_data(raw_data, channels, sample_interval, bp_filters)
            
            # 收集此受试者的训练数据块数据
            subject_train_data_list = []
            subject_train_labels_list = []
            
            for train_block in all_blocks:
                subject_train_data_list.append(processed_data[:, :, :, :, train_block])
                labels = np.repeat(np.arange(config['totalcharacter']), processed_data.shape[3])
                subject_train_labels_list.append(labels)
            
            # 合并受试者训练数据
            subject_train_data = np.concatenate(subject_train_data_list, axis=3)
            subject_train_labels = np.concatenate(subject_train_labels_list)
            
            # 重塑数据以适配模型输入
            subject_train_data = subject_train_data.transpose(3, 2, 0, 1)
            
            # 加载全局模型权重
            model.load_state_dict(torch.load('global_model_block_{}.pth'.format(block+1)))
            
            # 微调模型
            model = train_model_second_stage(model, subject_train_data, subject_train_labels, config, device)
            
            # 在测试数据块上评估
            test_data = processed_data[:, :, :, :, block]
            test_data = test_data.transpose(3, 2, 0, 1)  # 重塑以适配模型输入
            
            # 创建测试标签
            test_labels = np.repeat(np.arange(config['totalcharacter']), test_data.shape[0] // config['totalcharacter'])
            
            # 评估模型
            accuracy, predictions = evaluate_model(model, test_data, test_labels, device)
            acc_matrix[subject-1, block] = accuracy
            
            # 更新混淆矩阵
            true_labels = test_labels
            conf_matrix = confusion_matrix(true_labels, predictions, labels=np.arange(config['totalcharacter']))
            all_conf_matrix += conf_matrix
            
            print("    受试者 {} 准确率: {:.4f}".format(subject, accuracy))
        
        # 保存此数据块的混淆矩阵
        np.save('confusion_matrix_block_{}.npy'.format(block+1), all_conf_matrix)
    
    # 保存准确率矩阵
    np.save('accuracy_matrix.npy', acc_matrix)
    
    # 计算并保存ITR
    total_time = config['visual_cue'] + signal_length
    itr_matrix = calculate_itr(acc_matrix, config['totalcharacter'], total_time)
    np.save('itr_matrix.npy', itr_matrix)
    
    # 打印摘要统计信息
    print("\n" + "="*50)
    print("摘要统计信息")
    print("="*50)
    print("数据集: {}".format(dataset_type))
    print("信号长度: {} 秒".format(signal_length))
    print("使用通道数: {}".format(len(channels)))
    print("平均准确率: {:.4f} ± {:.4f}".format(np.mean(acc_matrix), np.std(acc_matrix)))
    print("平均ITR: {:.2f} ± {:.2f} bits/min".format(np.mean(itr_matrix), np.std(itr_matrix)))
    print("="*50)


if __name__ == "__main__":
    main()