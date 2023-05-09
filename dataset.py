import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder

# 表示数值特征从第42列开始
continous_features = 41

# Load and concatenate training data
train_files = glob.glob("../recsys2023_data/train/*.csv")  # Match all CSV files in the 'train' folder
train_data = pd.concat([pd.read_csv(file, sep='\t') for file in train_files])
# train_data = pd.read_csv(train_files[0],sep='\t') # choose first csv to train test

# Split into features and labels
X_train = train_data.iloc[:, 1:-2]  # Exclude the first column (RowId) and the last two columns (labels)
y_click = train_data["is_clicked"]
y_install = train_data["is_installed"]

# Get test data
test_data = pd.read_csv("../recsys2023_data/test/000000000000.csv",sep='\t')
X_test = test_data.iloc[:,1:] # Remove RowId


# 将所有数据组合在一起，对数据进行预处理（包括将离散值重新编码）和归一化
all_data = pd.concat([X_train,X_test])
df_cate = all_data.iloc[:,0:continous_features]
df_cate.fillna(df_cate.mode().iloc[0],inplace=True)
lbe = LabelEncoder() # 对离散特征进行编码
for i in range(41):
    df_cate[df_cate.columns[i]] = lbe.fit_transform(df_cate[df_cate.columns[i]])

df_value = all_data.iloc[:,continous_features:]
df_value.fillna(df_value.mean(),inplace=True)
df_value = (df_value - df_value.mean()) / df_value.std()

all_data = pd.concat([df_cate,df_value],axis=1)

# 得到feature_sizes
feature_sizes = []
for i in range(79):
    size = all_data.iloc[:,i].value_counts().shape[0]
    feature_sizes.append(size)
# 还原回去
n = train_data.shape[0]
X_train = all_data.iloc[:n,:]
X_test = all_data.iloc[n:,:]


# 定义数据类
class RecSysDataset(Dataset):
    
    def __init__(self, train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - train: Train or test. Required.
        """
        self.train = train

        if self.train:
            self.train_data = np.array(X_train)
            self.target = np.array(y_click)
        else:
            self.test_data = np.array(X_test)
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            # index of continous features are zero
            Xi_coutinous = np.zeros_like(dataI[continous_features:])
            Xi_categorial = dataI[:continous_features]
            Xi = torch.from_numpy(np.concatenate((Xi_categorial, Xi_coutinous)).astype(np.int32)).unsqueeze(-1)
            
            # value of categorial features are one (one hot features)
            Xv_categorial = np.ones_like(dataI[:continous_features])
            Xv_coutinous = dataI[continous_features:]
            Xv = torch.from_numpy(np.concatenate((Xv_categorial, Xv_coutinous)))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            # index of continous features are zero
            Xi_coutinous = np.ones_like(dataI[continous_features:])
            Xi_categorial = dataI[:continous_features]
            Xi = torch.from_numpy(np.concatenate((Xi_categorial, Xi_coutinous)).astype(np.int32)).unsqueeze(-1)
            
            # value of categorial features are one (one hot features)
            Xv_categorial = np.ones_like(dataI[:continous_features])
            Xv_coutinous = dataI[continous_features:]
            Xv = torch.from_numpy(np.concatenate((Xv_categorial, Xv_coutinous)))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

