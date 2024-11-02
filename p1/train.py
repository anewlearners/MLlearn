import numpy as np
import math
#读写数据
import pandas as pd
import os
import csv
#进度条
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

#设置随机种子
def same_seed(seed):
    torch.backends.cudnn.deterministic = True ##保证确定的卷积算法
    torch.backends.cudnn.benchmark = False  ##不随机使用卷积算法

    np.random.seed(seed)#np的种子保持一致
    torch.manual_seed(seed)#cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)##gpu



##划分数据集train vali test
def train_vail_split(data_set,valid_ratio,seed):
    valid_dara_size = int(len(data_set)*valid_ratio)
    train_data_size = len(data_set) - valid_dara_size
    trian_data,valid_data = random_split(data_set,
                                         lengths=[train_data_size,valid_dara_size],
                                         generator=torch.Generator().manual_seed(seed)
                                         )
    return np.array(trian_data),np.array(valid_data)


#选择特征
def select_feet(train_data,valid_data,test_data,select_all = True):
    y_train = train_data[:,-1]#选择最后一列，也就是label
    y_valid= valid_data [:,-1]
    #选择fature
    raw_x_train = train_data[:,:-1]#选择除了最后一列
    raw_x_valid = valid_data[:,:-1]

    raw_x_test = test_data
    if select_all:
        feat_id = list(range(raw_x_train.shape[1]))#fature 117
    else:
        feat_id = [0,1,2,3]
    return raw_x_train[:,feat_id], raw_x_valid[:,feat_id],raw_x_test[:,feat_id],y_train,y_valid



#数据集
class COVID19Dataset(Dataset):
    def __init__(self,feature,targets = None):
        if targets is None:
            self.targets = targets
        else:
            self.targets = torch.FloatTensor(targets)#标签
        self.features = torch.FloatTensor(feature)#特征张量
    def __getitem__(self,idx):
        if self.targets is None:#做预测
            return self.features[idx]
        else:#训练
            return self.features[idx],self.targets[idx]
    def __len__(self):
        ##返回数据集长度
        return len (self.features)#h行数


#module
class My_Module(nn.Module):
    def __init__(self,input_dim):
        super(My_Module,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
        )
    def forward(self,x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

##参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
#字典
config = {
    'seed':5201314,
    'selsec_all': True,
    'valid_ratio':0.2,
    'n_epochs' : 3000,
    'batch_size':256,
    'lr': 1e-5,
    'early_stop': 300,
    'save_path': '/home/panjunzhong/MLlearn/p1/models/model.ckpt'
}


def trainer(train_loader,valid_loader,config, device,model):
    criterion = nn.MSELoss(reduction= 'mean')
    optimizer = torch.optim.SGD(model.parameters(),lr = config['lr'],momentum = 0.9)
    writer = SummaryWriter()
    if not os.path.isabs('/home/panjunzhong/MLlearn/models'):
        os.mkdir('/home/panjunzhong/MLlearn/models')
    n_epochs = config['n_epochs']
    best_loss = math.inf #初始值无穷大
    step = 0 #记录当前训练多少伦次数据
    early_stop_count = 0 #没改变就加一，改变清零，可以提前终止
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        #进度条
        train_pbar = tqdm(train_loader,position  = 0 ,leave = True)
        for x,y in train_pbar:
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss=  criterion(pred,y)
            loss.backward() 
            optimizer.step()#优化模型
            step += 1
            loss_record.append(loss.detach().item())#保存数据


            train_pbar.set_description(f'Epoch[{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss':loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        #折线图
        writer.add_scalar(tag='Loss/train',scalar_value=mean_train_loss, global_step=step)


        #valid loop
        model.eval()
        loss_record_valid = []
        for x,y in valid_loader:
            x,y = x.to(device),y.to(device)
            with torch.no_grad():
                pre =  model(x)
                loss = criterion(pre,y)
            loss_record_valid.append(loss.item())
        mean_valid_loss = sum(loss_record_valid)/len(loss_record_valid)
        print(f'Epoch [{epoch+1} / {n_epochs}]: Train loss :{mean_train_loss :.4f},Valid loss {mean_valid_loss:.4f}')
        #可视化
        writer.add_scalar(tag = 'Loss/Valid',scalar_value = mean_valid_loss,global_step = step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),config['save_path'])
            print(f'saving model with loss {best_loss:.3f}')
            early_stop_count = 0
        else:
            early_stop_count+=1
        if early_stop_count > config['early_stop']:
            print('\n MOdel is not improving') 
            return

#预测
def predict(test_loader, model ,device):
    model.eval()
    preds  = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds,dim= 0).numpy()
    return preds



def save_pre(preds,file):
    with open(file,'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id','tested_positive'])
        for i ,p in enumerate(preds):
            writer.writerow([i,p])
        

if __name__ == "__main__":
    same_seed = (config['seed'])
    #读数据
    train_data = pd.read_csv('/home/panjunzhong/MLlearn/p1/covid.train.csv').values
    test_data  = pd.read_csv('/home/panjunzhong/MLlearn/p1/covid.test.csv').values

    #划分数据集
    train_data,valid_data = train_vail_split(train_data,config['valid_ratio'],same_seed)
    print(f'train_data size{train_data.shape}, valid_data size {valid_data.shape}, test_daata size {test_data.shape} ')
    #选择特征
    x_train,x_valid,x_test,y_train,y_valid = select_feet(train_data,
    valid_data, test_data,same_seed)


    #构造数据集
    train_dataset = COVID19Dataset(x_train,y_train)
    valid_dataset = COVID19Dataset(x_valid,y_train)
    test_dataset = COVID19Dataset (x_test)

    #Dataloader
    train_loader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle = True,pin_memory = True)
    valid_loader = DataLoader(valid_dataset,batch_size=config['batch_size'],shuffle = True,pin_memory = True)
    test_loader  = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle = False,pin_memory = True)

    #训练
    model = My_Module(input_dim=train_data.shape[1] - 1).to(device)
    trainer(train_loader , valid_loader, config,device,model)
    
    
    preds = predict(test_loader,model,device)
    save_pre(preds,file = '/home/panjunzhong/MLlearn/p1/models/pred.csv' )
    
    
    
  
    


             



         




