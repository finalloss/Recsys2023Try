import torch.optim as optim
from torch.utils.data import DataLoader

from DeepFM import DeepFM
from dataset import RecSysDataset
from args import feature_sizes

# load data
train_data,val_data,test_data = RecSysDataset(train=1),RecSysDataset(train=2),RecSysDataset(train=0)
loader_train = DataLoader(train_data, batch_size=512,shuffle=True)
loader_val = DataLoader(val_data, batch_size=512,shuffle=False)
loader_test = DataLoader(test_data,batch_size=512)
print("Load Data finished...")

# training
model = DeepFM(feature_sizes, use_cuda=True,cuda_name='cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
print("Start Training...")
model.fit(loader_train, loader_val, optimizer, epochs=100, verbose=True,print_every=200)
model.Get_result(loader_test,model)
