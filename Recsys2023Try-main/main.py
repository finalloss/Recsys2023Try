import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DeepFM import DeepFM
from dataset import RecSysDataset
import pandas as pd
from args import feature_sizes

# load data
train_data,val_data,test_data = RecSysDataset(train=1),RecSysDataset(train=2),RecSysDataset(train=0)
loader_train = DataLoader(train_data, batch_size=512,shuffle=True)
loader_val = DataLoader(val_data, batch_size=512,shuffle=False)
loader_test = DataLoader(test_data,batch_size=512)
print("Load Data finished...")

# batch_size和activation可以有不同选择
# training
# 当筛选特征之后需要相应地修改nume_fea_size
model = DeepFM(feature_sizes, embedding_size=4,nume_fea_size=16,
               hidden_dims=[64,64,64],num_classes=1,dropout=[0.3,0.3,0.3],
               use_cuda=True,cuda_name='cuda:0')
# weight_decay 权重衰减系数 L2正则化项，防止过拟合，也可能导致模型训练效果下降
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
schedule = ReduceLROnPlateau(optimizer,'min',factor=0.2,patience=4,min_lr=1e-6,verbose=True)
print("Start Training...")
# wait用于早停策略
lab_id = 25
model.fit(loader_train, loader_val, optimizer,schedule, epochs=100, 
          verbose=True,print_every=500,wait=12,lrd=True,figure_num=lab_id)
model.Get_result(loader_test,model)

# 得到输出的结果
# test_data = pd.read_csv("../recsys2023_data/test/000000000000.csv",sep='\t')
# RowId = test_data.iloc[:,0]
# i = 0
# with open('./Result/click_result.txt', 'r') as f1, \
# open('./Result/install_result'+str(lab_id)+'.txt', 'r') as f2,\
# open('./Finish/result' + str(lab_id) + '.txt', 'w') as r:
#     r.write("RowId\tis_clicked\tis_installed\n")
#     for line1, line2 in zip(f1, f2):
#         # 将 line1 和 line2 整合成新的一行
#         r.write(str(RowId[i])+'\t')
#         r.write(line1.strip() +'\t'+ line2.strip() + '\n')
#         i+=1

