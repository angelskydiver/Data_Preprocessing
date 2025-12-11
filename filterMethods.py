##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           在模拟过程数据上实现过滤法
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 读取数据
import numpy as np
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

#%% 分离 X 和 y
y = VSdata[:,0]
X = VSdata[:,1:]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           基于线性相关的变量选择
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 计算基于线性相关的得分
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

VSmodel_Correlation = SelectKBest(f_regression, k=10).fit(X, y)
input_scores = VSmodel_Correlation.scores_

# 找出排名靠前的输入变量
top_k_inputs_Correlation = np.argsort(input_scores)[::-1][:10] + 1#  [::-1] 将 argsort() 返回的数组反转，[:n] 取前 n 个元素
print(top_k_inputs_Correlation)

# 将 X 缩减为仅包含排名靠前的相关输入变量
X_relevant = VSmodel_Correlation.transform(X)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           基于互信息（MI）的变量选择
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 计算基于互信息的得分
from sklearn.feature_selection import mutual_info_regression

VSmodel_MI = SelectKBest(mutual_info_regression, k=10).fit(X, y)
input_scores = VSmodel_MI.scores_

# 找出排名靠前的输入变量
top_k_inputs_MI = np.argsort(input_scores)[::-1][:10] #  [::-1] 将 argsort() 返回的数组反转，[:n] 取前 n 个元素
print(top_k_inputs_MI)

# 将 X 缩减为仅包含排名靠前的相关输入变量
X_relevant = VSmodel_MI.transform(X)