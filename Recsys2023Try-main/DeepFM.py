import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from args import click_feature


class DeepFM(nn.Module):
    
    def __init__(self, feature_sizes, embedding_size=8,nume_fea_size=38,
                 hidden_dims=[64, 64, 64], num_classes=1, dropout=[0.3, 0.3, 0.3], 
                 use_cuda=True,cuda_name="cuda:0"):
        """
        Initialize a new network

        Inputs: 
        - feature_size: A list of integer giving the size of features for each category field.
        - embedding_size: An integer giving size of feature embedding.
        - nume_fea_size: num of dense features
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - cuda_name: which cuda use 
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.nume_fea_size = nume_fea_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long
        self.cuda_name = cuda_name
        self.activation = nn.ReLU()
        
        #  check if use cuda
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device(self.cuda_name)
        else:
            self.device = torch.device('cpu')

        # init fm part
        self.fm_first_order_dense = nn.Linear(self.nume_fea_size,1) # 数值特征的一阶表示
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes]) # 类别特征的一阶表示
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]) # 类别特征的二阶表示

        # init deep part
        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims
        self.all_dims = all_dims
        self.dense_linear = nn.Linear(self.nume_fea_size,self.field_size * embedding_size) # 数值特征的维度变换到与FM输出维度一致
        for i in range(1, len(hidden_dims) + 1):
            # setattr是内置函数，用于动态设置对象的属性 setattr(obj, name, value)
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self,'activation_' + str(i),self.activation)
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))
            
        self.dnn_linear = nn.Linear(all_dims[-1],num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [batch_size, cate_fea_size]
        X_dense: 数值型特征输入  [bs, dense_fea_size]
        """
        
        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1, keepdim=True)  # [bs, 1]
        
        if X_dense is not None:
            fm_1st_dense_res = self.fm_first_order_dense(X_dense) 
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]
        
        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)
        
        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2 
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        
        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]
        
        if X_dense is not None:
            dense_out = self.activation(self.dense_linear(X_dense))   # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out   # [bs, n * emb_size]
        
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        
        dnn_out = self.dnn_linear(dnn_out)   # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
        out = self.sigmoid(out)
        return out.squeeze()

    def fit(self, loader_train, loader_val, optimizer,schedule, epochs=100, verbose=True, print_every=100, wait=8,lrd=True,figure_num=1):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: train_data
        - loader_val: val_data
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - schedeule: to reduce lr
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        - wait: if val_loss after wait iteration not down, break
        - lrd: Bool, if reduce lr
        """
        
        # load input data
        self.figure_num = figure_num
        model = self.train().to(device=self.device)
        self.model_path = "./models/" + "lab" + str(figure_num) + "best_weights.pth"
        # 尝试对正样本的误差给予更多关注
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(self.device))
        criterion = nn.BCELoss()
        if lrd:
            schedule = schedule
        train_loss = []
        val_loss = []
        # 记录每print_every次迭代后的平均loss
        loss_train = []
        loss_val = 0
        # 记录每print_every次迭代后的accuracy
        num_correct = 0
        num_samples = 0

        min_val_loss = 1e5

        for epoch in range(epochs):
            pbar = tqdm(loader_train)
            for i,data in enumerate(pbar):
                xi,xv,y=data
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)
                
                total = model(xi, xv)
                loss = criterion(total, y)
                loss_train.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'iteration':i,'train loss':loss.item()})
                
                if (i) % print_every == 0:
                    train_loss_one = sum(loss_train)/len(loss_train)
                    train_loss.append(train_loss_one)
                    with torch.no_grad():
                        for xi, xv, y in loader_val:
                            xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                            xv = xv.to(device=self.device, dtype=torch.float)
                            y = y.to(device=self.device, dtype = torch.float)
                            total = model(xi, xv)

                            preds = total > 0.5
                            num_correct += (preds == y).sum()
                            num_samples += preds.size(0)

                            # 尝试为sklearn中的log_loss
                            # loss = log_loss(y.cpu().detach().numpy(),total.cpu().detach().numpy(),labels=[0,1],eps=1e-7,normalize=True)
                            # loss_val += loss
                            loss = criterion(total,y)
                            loss_val += loss.item()
                        
                    loss_val = loss_val/len(loader_val)
                    val_loss.append(loss_val)
                    if verbose:
                        print("\nEpoch{}----Iteration{}----train loss:{:10.6f}   val loss:{:10.6f}".
                          format(epoch,i,train_loss_one, loss_val))

                    if lrd:
                        schedule.step(loss_val)

                    if loss_val < min_val_loss:
                        min_val_loss = loss_val
                        acc = float(num_correct / num_samples)
                        print("Updata min_val_loss to {:.6f},val_acc = {:5.4f}"
                        .format(min_val_loss, acc))
                        delay = 0
                        self.min_val_loss = min_val_loss
                        self.acc = acc
                        torch.save(model.state_dict(),self.model_path)
                    else:
                        delay = delay + 1
                    if delay > wait:
                        break

                    loss_train = []
                    loss_val = 0
                    num_correct = 0
                    num_samples = 0
            if delay > wait:
                self.epoch = epoch
                break
        self.plot_loss(train_loss,val_loss)

    
    def plot_loss(self,train_loss,val_loss):
        plt.figure()
        plt.plot(train_loss,c='red',label= 'train_loss')
        plt.plot(val_loss, c="blue", label="val_loss")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("CrossEntropyLoss")
        plt.title("CrossEntropyLoss of Train and Validation in each Iteration")
        if click_feature:
            str1 = "./loss_pictures/click_loss"+str(self.figure_num)+".png"
            plt.savefig(str1)
        else:
            str2 = "./loss_pictures/install_loss"+str(self.figure_num)+".png"
            plt.savefig(str2)
        print("Epoch = {},最终min_val_loss = {:.6f},Accuracy = {:.4f} ".format(self.epoch,self.min_val_loss,self.acc))


    def Get_result(self,loader,model):
        model.load_state_dict(torch.load(self.model_path))
        if click_feature:
            file_name = './Result/click_result'+str(self.figure_num)+'.txt'
        else:
            file_name = './Result/install_result'+str(self.figure_num)+'.txt'
        with torch.no_grad():
            with open(file_name,'w') as f:
                for xi,xv in loader:
                    xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                    xv = xv.to(device=self.device, dtype=torch.float)
                    preds = model(xi,xv)
                    for value in preds.cpu().numpy():
                        f.write(str(value)+ '\n')
        


