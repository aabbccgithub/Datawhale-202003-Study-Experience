# Task2 数据分析心得
## --Datawhale零基础入门数据挖掘
### 1.概述
本次为做比赛、做研究，拿到数据后的第一步：数据探索性分析：Exploratory Data Analysis。本次主要基础是学习运用数据3个数据科学库：pandas、numpy和scipy。2个可视化库matplotlib和seaborn。在数据方便包括数据读入、观察、相关数值特征的查看、缺失值异常值处理、特征分布、特征相关性关系探索。
### 2.细分
#### 2.1 pandas
pandas可以用来读取、存入数据，对数据有着很多原生的函数可以用。
#### 2.2 numpy
numpy可以用来快速的做矩阵/线性代数方面的运算，科学计算非常快。
#### 2.3 scipy
scipy有很多统计方面的包，非常值得一用。
#### 2.4 matplotlib
matplotlib可以从头开始画图，画图必备工具。
#### 2.5 seaborn
seaborn有很多集成好了的函数用于画图，画出的图很漂亮、快速。
#### 2.6 数据读入与观察
用pandas.read_csv，可以读入csv与文本文件。
data.shape：可以查看数据的形状大小。
很重要的一点是：最好结合特征的具体含义（如果已知特征的相关信息），一定要充分应用上去，对于解决实际问题非常有效。
#### 2.7 相关数值特征的查看
data.describe()：可以查看数据的样本数、均值、标准差、最小值分位数、最大值等。
data.info()：可以查看变量特征的类型与非缺失的样本数。
#### 2.8 缺失值异常值处理
data.isnull().sum()：可以查看缺失的nan总数。
data.plot.bar()：可以用来画柱状图。
msno.matrix(data)：可以可视化查看缺失值的位置。
#### 2.9 特征分布
可以用scipy.stats中的st结合sns.distplot，用理论分布拟合经验分布。可视化操作，非常有效。
skewness and kurtosis：偏度和峰度，data.skew()、data.kurt()主要用于衡量其与正态分布的差异。
plt.hist：分布直方图
分类特征：categorical feature：主要统计频数，用unique；data.value_counts()可以统计频数。
#### 2.10 特征相关性关系探索
data.corr()：可以计算特征之间的相关系数。
可以用sns.heatmap()画出相关系数的热图。
sns.pairplot()：这个函数极其有用，可以用来探究特征之间两两关系。
