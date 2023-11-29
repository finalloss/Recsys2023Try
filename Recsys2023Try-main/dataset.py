from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from args import continous_features,click_feature,alldata_feature

# Load and concatenate training data
train_files = glob.glob("../recsys2023_data/train/*.csv")  # Match all CSV files in the 'train' folder
if alldata_feature:
    train_data = pd.concat([pd.read_csv(file, sep='\t') for file in train_files])
else:
    train_data = pd.read_csv(train_files[0],sep='\t') # choose first csv to train test

# Split into features and labels
X_train = train_data.iloc[:, 1:-2]  # Exclude the first column (RowId) and the last two columns (labels)
y_click = train_data["is_clicked"]
y_install = train_data["is_installed"]

# Get test data
test_data = pd.read_csv("../recsys2023_data/test/000000000000.csv",sep='\t')
X_test = test_data.iloc[:,1:] # Remove RowId

# 确定筛选特征
features_name = ['f_2', 'f_3', 'f_4', 'f_6', 'f_8', 'f_10', 'f_11', 'f_12', 'f_14',
       'f_16', 'f_17', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
       'f_32', 'f_34', 'f_35', 'f_37', 'f_40', 'f_41', 'f_42', 'f_48', 'f_49',
       'f_50', 'f_51', 'f_55', 'f_56', 'f_57', 'f_58', 'f_59', 'f_65', 'f_68',
       'f_69', 'f_72', 'f_78', 'f_79']
X_train = X_train[features_name]
X_test = X_test[features_name]


# 将所有数据组合在一起，对数据进行预处理（包括将离散值重新编码）和归一化
all_data = pd.concat([X_train,X_test])
df_cate = all_data.iloc[:,0:continous_features]
df_cate.fillna(df_cate.mode().iloc[0],inplace=True)
lbe = LabelEncoder() # 对离散特征进行编码
for i in range(continous_features):
    df_cate[df_cate.columns[i]] = lbe.fit_transform(df_cate[df_cate.columns[i]])

df_value = all_data.iloc[:,continous_features:]
df_value.fillna(df_value.mean(),inplace=True)
df_value = (df_value - df_value.mean()) / df_value.std()

all_data = pd.concat([df_cate,df_value],axis=1)

# 得到feature_sizes
feature_sizes = []
for i in range(X_train.shape[1]):
    size = all_data.iloc[:,i].value_counts().shape[0]
    feature_sizes.append(size)

# 还原回去
n = train_data.shape[0]
X_train = all_data.iloc[:n,:]
X_test = all_data.iloc[n:,:]

# 划分训练集和验证集 
if click_feature:
    train_x,val_x,train_y,val_y = train_test_split(X_train,y_click,test_size=0.2,stratify=y_click)
else:
    train_x,val_x,train_y,val_y = train_test_split(X_train,y_install,test_size=0.2,stratify=y_install)

# 定义数据类
class RecSysDataset(Dataset):
    
    def __init__(self, train=1):
        """
        Initialize file path and train/test mode.

        Inputs:
        - train: 0 test_data    1 train_data    2 validate_data. Required.
        - Xi: category features
        - Xv: dense features
        """
        self.train = train

        if self.train:
            if self.train == 1:
                self.train_data = np.array(train_x)
                self.target = np.array(train_y)
            else:
                self.train_data = np.array(val_x)
                self.target = np.array(val_y)
        else:
            self.test_data = np.array(X_test)
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = dataI[:continous_features]
            Xv = dataI[continous_features:]
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            Xi = dataI[:continous_features]
            Xv = dataI[continous_features:]
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

