##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    过程信号去噪
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 读取数据
import numpy as np
noisy_signal = np.loadtxt('noisy_flow_signal.csv', delimiter=',')

#%% SMA 滤波
import pandas as pd

windowSize = 15
smoothed_signal_MA = pd.DataFrame(noisy_signal).rolling(windowSize).mean().values

#%% SG 滤波
from scipy.signal import savgol_filter

smoothed_signal_SG = savgol_filter(noisy_signal, window_length = 15, polyorder = 2)

#%% 绘图
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei']

plt.figure(figsize=(11,3))
plt.plot(noisy_signal, alpha=0.3, label='含噪信号')
plt.plot(smoothed_signal_MA, color='m', label='SMA 平滑信号')
plt.plot(smoothed_signal_SG, color='orange', label='SG 平滑信号')
plt.xlabel('样本编号'), plt.ylabel('数值')
plt.legend()
plt.show()