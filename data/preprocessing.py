"""
SSVEP BCI的数据预处理模块。
"""
import numpy as np
from scipy.signal import butter, filtfilt
from .config import CHANNELS_9, SUBBAND_NUM, FILTER_ORDER, HIGH_CUTOFF, LOW_CUTOFF_BASE


def create_bandpass_filters(sampling_rate, subband_num=SUBBAND_NUM):
    """
    为子带预处理创建带通滤波器。
    
    参数:
        sampling_rate (int): 信号的采样率
        subband_num (int): 子带数量
        
    返回:
        list: 滤波器系数列表
    """
    filters = []
    high_cutoff = HIGH_CUTOFF
    low_cutoff_base = LOW_CUTOFF_BASE
    
    for i in range(1, subband_num + 1):
        low_cutoff = low_cutoff_base * i
        # 归一化频率
        low = low_cutoff / (sampling_rate / 2)
        high = high_cutoff / (sampling_rate / 2)
        
        # 创建巴特沃斯带通滤波器
        b, a = butter(FILTER_ORDER, [low, high], btype='band')
        filters.append((b, a))
    
    return filters


def preprocess_data(data, channels=CHANNELS_9, sample_interval=None, bp_filters=None):
    """
    使用带通滤波器预处理原始EEG数据。
    
    参数:
        data (np.ndarray): 原始EEG数据
        channels (list): 要使用的通道索引列表
        sample_interval (slice): 要提取的样本区间
        bp_filters (list): 要应用的带通滤波器
        
    返回:
        np.ndarray: 预处理后的数据
    """
    if sample_interval is not None:
        # 从指定通道和信号区间提取数据
        sub_data = data[channels, sample_interval, :, :]
    else:
        sub_data = data[channels, :, :, :]
    
    total_channels = len(channels)
    sample_length = sub_data.shape[1]
    subband_num = len(bp_filters) if bp_filters else SUBBAND_NUM
    total_character = sub_data.shape[2]
    total_block = sub_data.shape[3]
    
    # 初始化处理后的数据数组
    processed_data = np.zeros((total_channels, sample_length, subband_num, total_character, total_block))
    
    # 应用带通滤波器
    for char in range(total_character):
        for blk in range(total_block):
            tmp_raw = sub_data[:, :, char, blk]
            for i, (b, a) in enumerate(bp_filters):
                filtered_signal = np.zeros((total_channels, sample_length))
                for ch in range(total_channels):
                    filtered_signal[ch, :] = filtfilt(b, a, tmp_raw[ch, :])
                processed_data[:, :, i, char, blk] = filtered_signal
                
    return processed_data