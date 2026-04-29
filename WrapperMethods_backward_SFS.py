##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           在模拟过程数据上实现后向 SFS
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 读取数据
import numpy as np
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

#%% 分离 X 和 y
y = VSdata[:,0]
X = VSdata[:,1:]

#%% 数据标准化
from sklearn.preprocessing import StandardScaler
xscaler = StandardScaler()
X_scaled = xscaler.fit_transform(X)

yscaler = StandardScaler()
y_scaled = yscaler.fit_transform(y[:,None])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##           基于 SFS 的变量选择
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

BSFS = SequentialFeatureSelector(LinearRegression(), n_features_to_select=10, direction='backward', cv=5).fit(X_scaled, y_scaled)

#%% 查看已选输入变量
print('已选输入变量: ', BSFS.get_support(indices=True)+1) # 返回所选特征的整数索引

#%% 将 X 缩减为仅包含相关输入变量
X_relevant = BSFS.transform(X)