import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from IPython.display import display, clear_output
import time
x = np.linspace(0,5)
f, ax = plt.subplots()
ax.set_title("Bessel functions")

for n in range(1,10):
    time.sleep(1)
    ax.plot(x, jn(x,n))
    clear_output(wait=True)
    display(f)

# close the figure at the end, so we don't get a duplicate
# of the last plot
plt.close()


## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time
from tqdm import tqdm
import itertools

warnings.filterwarnings('ignore')
%matplotlib inline

## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import scipy.signal as signal

#处理异常值
def smooth_cols(group,cols = ['power'],out_value = 600):
    for col in cols:
        yes_no = (group[col]<out_value).astype('int')
        new = yes_no * group[col]
        group[col] = new.replace(0,group[col].median())
    return group
def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]

#定义日期提取函数
def date_tran(df,fea_col):
    for f in tqdm(fea_col):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return (df)

### count编码
def count_coding(df,fea_col):
    for f in tqdm(fea_col):
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return(df)
#定义交叉特征统计
def cross_cat_num(df,num_col,cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median', '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_std'.format(f1, f2): 'std', '{}_{}_mad'.format(f1, f2): 'mad',
            })
            df = df.merge(feat, on=f1, how='left')
    return(df)


## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
Train_data = pd.read_csv('datalab/231784/used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('datalab/231784/used_car_testA_20200313.csv', sep=' ')

#Train_data = Train_data[Train_data['price']>100]
Train_data['price'] = np.log1p(Train_data['price'])
## 输出数据的大小信息
print('Train data shape:',Train_data.shape)
print('TestA data shape:',TestA_data.shape)


#合并数据集
concat_data = pd.concat([Train_data,TestA_data])
concat_data.index = range(len(concat_data))
concat_data = concat_data.groupby('brand').apply(smooth_cols,cols = ['power'],out_value = 600)
concat_data.index = range(len(concat_data))

concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-',0) 
print('concat_data shape:',concat_data.shape)


#提取日期信息
date_cols = ['regDate', 'creatDate']
concat_data = date_tran(concat_data,date_cols)

# 对类别较少的特征采用one-hot编码
one_hot_list = ['fuelType','gearbox','notRepairedDamage']
for col in one_hot_list:
    one_hot = pd.get_dummies(concat_data[col])
    one_hot.columns = [col+'_'+str(i) for i in range(len(one_hot.columns))]
    concat_data = pd.concat([concat_data,one_hot],axis=1)


data = concat_data.copy()
data = data.fillna(data.mode().iloc[0,:])

#count编码
count_list = ['regDate', 'creatDate', 'model', 'brand', 'regionCode','bodyType','fuelType','gearbox','notRepairedDamage','name']
data = count_coding(data,count_list)

# 计算某品牌的销售统计量，同学们还可以计算其他特征的统计量
# 这里要以 train 的数据计算统计量
Train_gb = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_ptp'] = kind_data.price.ptp()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
data = data.merge(brand_fe, how='left', on='brand')

### 用数值特征对类别特征做统计刻画，随便挑了几个跟price相关性最高的匿名特征
cross_cat = ['model', 'brand', 'regionCode']
cross_num = ['v_0', 'v_3', 'v_8', 'v_12']
data = cross_cat_num(data,cross_num,cross_cat)


#特征构造
# 使用时间：data['creatDate'] - data['regDate']，反应汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要 errors='coerce'
data['used_time1'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days/30

## 选择特征列
numerical_cols = data.columns
#print(numerical_cols)

cat_fea = ['SaleID','offerType','seller']
feature_cols = [col for col in numerical_cols if col not in cat_fea]
feature_cols = [col for col in feature_cols if col not in ['price']]

## 提前特征列，标签列构造训练样本和测试样本
X_data = data.iloc[:len(Train_data),:][feature_cols]
Y_data = Train_data['price']
X_test  = data.iloc[len(Train_data):,:][feature_cols]

#删除已经编码的特征
drop_list = one_hot_list + count_list
X_data = X_data.drop(drop_list,axis=1)
X_test = X_test.drop(drop_list,axis=1)


## 定义了一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))


## 绘制标签的统计图，查看标签分布
plt.hist(Y_data)
plt.show()
plt.close()


def build_model_lgb(x_train,y_train):
    gbm = lgb.LGBMRegressor(n_estimators=1000,gamma=0, subsample=0.7,\
        colsample_bytree=0.9, max_depth=7,feature_fraction=0.9)
    param_grid = {
        'learning_rate': [ 0.1,],
    }
    #gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm

x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.2,random_state=42)
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(np.expm1(y_val),np.expm1(val_lgb))
print('MAE of val with lgb:',MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
subA_lgb = np.expm1(model_lgb_pre.predict(X_test))
print('Sta of Predict lgb:')
Sta_inf(subA_lgb)

sub.to_csv('/home/tianchi/myspace/sub_lgb.csv',index=False)

