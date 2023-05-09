import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from DeepFM import DeepFM
from dataset import RecSysDataset

# set num items for training and valid
# Num_train_all = 3485852
Num_train_all = 116474 # 仅仅是第一个csv的数据，用于调试
Num_train = int(Num_train_all * 0.9)

# load data
train_data = RecSysDataset(train=True)
test_data = RecSysDataset(train=False)
loader_train = DataLoader(train_data, batch_size=128,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = RecSysDataset(train=True)
loader_val = DataLoader(val_data, batch_size=128,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, Num_train_all)))
print("Load Data finished...")
loader_test = DataLoader(test_data,batch_size=128)

feature_sizes = [22, 136, 5, 633, 6, 5167, 1, 6, 7, 3, 24, 26, 329, 19, 5801, 10, 49, 901, 19, 55, 34, 
                 24, 4, 4, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
model = DeepFM(feature_sizes, use_cuda=True,cuda_name='cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
print("Start Training...")
model.fit(loader_train, loader_val, optimizer, epochs=1, verbose=True)
model.Get_result(loader_test,model)