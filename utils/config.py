"""
SSVEP BCI项目的配置文件。
"""

# 数据集配置
DATASET_CONFIGS = {
    'Bench': {
        'totalsubject': 35,  # 总受试者数
        'totalblock': 6,     # 总数据块数
        'totalcharacter': 40, # 总字符数
        'sampling_rate': 250, # 采样率
        'visual_latency': 0.14, # 视觉延迟
        'visual_cue': 0.5,    # 视觉提示
        'total_ch': 64,       # 总通道数
        'max_epochs': 1000,   # 最大训练轮数
        'dropout_second_stage': 0.6  # 第二阶段dropout率
    },
    'BETA': {
        'totalsubject': 70,   # 总受试者数
        'totalblock': 4,      # 总数据块数
        'totalcharacter': 40,  # 总字符数
        'sampling_rate': 250,  # 采样率
        'visual_latency': 0.13, # 视觉延迟
        'visual_cue': 0.5,     # 视觉提示
        'total_ch': 64,        # 总通道数
        'max_epochs': 800,     # 最大训练轮数
        'dropout_second_stage': 0.7  # 第二阶段dropout率
    }
}

# 通道配置
CHANNELS_9 = [48, 54, 55, 56, 57, 58, 61, 62, 63]  # 9个通道的索引: (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2)
CHANNELS_64 = list(range(1, 65))  # 全部64个通道

# 模型配置
SUBBAND_NUM = 3  # 子带/带通滤波器数量
FILTER_ORDER = 2  # 带通滤波器的滤波器阶数
HIGH_CUTOFF = 90  # 所有带通滤波器的高截止频率
LOW_CUTOFF_BASE = 8  # 基础低截止频率 (第i个带通滤波器的低截止频率为8*i)