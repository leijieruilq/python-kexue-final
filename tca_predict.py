# In[]
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
from model.tcaffn.tcaffn import Model
import yaml
import warnings
import datetime  
warnings.filterwarnings('ignore')
plt.style.use('default')
from matplotlib import rcParams
config = {
    "font.family": 'SimHei', # 衬线字体
    "font.serif": ['SimHei'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
# In[]
def delete_file(file_path):  
    if os.path.exists(file_path):  
        try:  
            os.remove(file_path)  
            print(f"文件 {file_path} 已成功删除。")  
        except OSError as e:  
            print(f"删除文件 {file_path} 时出错: {e.strerror}")  
    else:  
        print(f"文件 {file_path} 不存在。")  
# In[]
def time_data_load(read_path):
    """
    Parameters
    ----------
    read_path : 导入的股票.xlsx的路径.

    Returns
    -------
    X : excel中的所有值.
    tab_cols : 时间序列列名称.
    date : date时间戳信息
    """
    tables=pd.read_excel(read_path)
    tab_cols=tables.columns
    X=tables.values
    date = tables['date']
    return tab_cols,X,date

# In[]
def choose_col_analysis(X,tab_cols,choose_channels):
    """
    Parameters
    ----------
    X : 导入的股票.xlsx的values.
    i : 第几列.
    tab_cols : 时间序列列名称.
    choose_channels : 选择的列号(从0开始)

    Returns
    -------
    X : X的第i列.
    tab_cols[i] : 第i列名称.
    """
    X = X[:,choose_channels]   
    X = X.astype(np.float32)
    return X,tab_cols[choose_channels]

# In[]
def normalization(x):
    """
    Parameters
    ----------
    x : 第i列的values.

    Returns
    -------
    x : 标准化后的x.
    means : 标准化均值.
    stds : 标准化的方差.
    """
    means = np.mean(x,axis=0)
    x = x - means
    stds = np.std(x,axis=0)
    x = x / stds
    return x,means,stds

# In[]
class Stamp_DataScaler():
    """
    Parameters
    ----------
    对时间戳进行标准化的data_Scaler类

    Returns
    -------
    标准化后的时间戳信息特征
    """
    def __init__(self):
        # Max Min Normalization
        self.max = 0
        self.min = 0

    def to(self, device):
        # self.to(device)
        self.max = self.max.to(device)
        self.min = self.min.to(device)
        return self
    
    def fit_trans(self, data):
        self.fit(data)
        return self.trans(data)
    
    def fit(self, data):
        self.max = data.max((0))[0].unsqueeze(0)
        self.min = data.min((0))[0].unsqueeze(0)
        for i in range(data.size(-1)):
            if self.max[...,i] == self.min[...,i]:
                self.max[...,i] = 0.0
                self.min[...,i] = self.max[...,i]-1

    def trans(self,data):
        data = (data - self.min)/(self.max-self.min)
        return data

    def inverse_transform(self, data, choise_channels):
        data = (self.max-self.min)*data + self.min
        return data
# In[]
def date_deal(date,use_for_test=False,stamp_scaler=None):
    """
    Parameters
    ----------
    date:输入的date_array

    Returns
    -------
    data_stamp:标准化后的时间戳信息特征(tensor类型)
    stamp_scaler:用于标准化时间戳信息的scaler(处理tensor类型的数据)
    """
    data_stamp = pd.DataFrame(date, columns=['date'])
    data_stamp['date'] = pd.to_datetime(data_stamp['date'])
    data_stamp['month'] = data_stamp['date'].apply(lambda row: row.month, 1)
    data_stamp['day'] = data_stamp['date'].apply(lambda row: row.day, 1)
    data_stamp['weekday'] = data_stamp['date'].apply(lambda row: row.weekday(), 1)
    data_stamp['hour'] = data_stamp['date'].apply(lambda row: row.hour, 1)
    data_stamp['minute'] = data_stamp['date'].apply(lambda row: row.minute, 1)
    data_stamp = data_stamp.drop(columns='date').values
    data_stamp = torch.tensor(data_stamp)
    if use_for_test==False:
        stamp_scaler = Stamp_DataScaler()
        stamp_scaler.fit(data_stamp)
        data_stamp = stamp_scaler.trans(data_stamp)
    else:
        """
        之后直接预测的时候可以使用.trans函数进行未来步时间戳信息的提取
        """
        data_stamp = stamp_scaler.trans(data_stamp)
    return data_stamp,stamp_scaler

# In[]
def generate_dataset(X,data_stamp,
                     num_timesteps_input, num_timesteps_output):
    """
    Parameters
    ----------
    X : 选中的values.
    data_stamp : time_stamp.
    num_timesteps_input : input的时间步数.
    num_timesteps_output : output的时间步数.

    Returns
    -------
    x:(batch_size,channels,1,num_input)
    y:(batch_size,channels,1,num_output) 
    x_mark:(batch_size,num_input,time_channels)
    y_mark:(batch_size,num_output,time_channels)
    """
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target, x_mark,y_mark = [], [], [] ,[]
    for i, j in indices:
        features.append(X[i: i + num_timesteps_input])
        target.append(X[i + num_timesteps_input: j])
        x_mark.append(data_stamp[i: i + num_timesteps_input].numpy())
        y_mark.append(data_stamp[i + num_timesteps_input:j].numpy())

    return torch.from_numpy(np.array(features)).permute(0,2,1).unsqueeze(dim=-2), \
           torch.from_numpy(np.array(target)).permute(0,2,1).unsqueeze(dim=-2),\
           torch.from_numpy(np.array(x_mark)),\
           torch.from_numpy(np.array(y_mark))


# In[]
def train_epoch(net,training_input,training_target,
                x_mark,y_mark,
                batch_size,loss_criterion,optimizer,device="cuda"):
    """
    Parameters
    ----------
    net : cnn-lstm.
    training_input : (batch,channels,1,num_input).
    training_target : (batch,channels,1,num_output).
    batch_size : batch.

    Returns
    -------
    training_loss
    """
    permutation = torch.randperm(training_input.shape[0])#随机排列顺序
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        x_seq_mark,y_seq_mark =  x_mark[indices], y_mark[indices]
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        x_seq_mark = x_seq_mark.to(device)
        y_seq_mark = y_seq_mark.to(device)

        out = net(X_batch,x_seq_mark,y_seq_mark)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)
# In[]
def val_evaluate(model,val_input,val_target,
                 x_mark,y_mark,
                 means,stds,loss_criterion,device="cuda"):
    """
    Parameters
    ----------
    model : torch.
    val_input : (b,c,n,t).
    val_target : (b,c,n,t).
    x_mark : (b,t,t_c)
    y_mark : (t,t,t_c)

    Returns
    -------
    val_loss,val_mae
    """
    with torch.no_grad():
        model.eval()
        val_input = val_input.to(device)
        val_target = val_target.to(device)
        x_mark = x_mark.to(device)
        y_mark = y_mark.to(device)
        """
        计算validation的loss
        """
        out = model(val_input,x_mark,y_mark)
        val_loss = loss_criterion(out, val_target).to(device="cpu")
        """
        计算validation的mae
        """
        out = out.detach().cpu().numpy()*stds + means  
        val_target = val_target.detach().cpu().numpy()*stds + means  
        mae = np.mean(np.absolute(out - val_target))
        mse = np.mean((val_target-out)**2)
        return mse,mae
def train(model,epoches,
          training_input,training_target,
          x_train_mark,y_train_mark,
          val_input,val_target,
          x_test_mark,y_test_mark,
          batch_size,
          loss_criterion,optimizer,
          means,stds,device,save_path):
    """
    Parameters
    ----------
    model : torch.
    epoches : train次数
    training_input : (batch,channels,n,num_input).
    training_target : (batch,channels,n,num_output).
    val_input : (batch,channels,n,num_input).
    val_target : (batch,channels,n,num_output).
    batch_size : 训练批次数

    Returns
    -------
    training_losses,validation_losses,validation_maes
    """
    training_losses = []
    validation_losses = []
    validation_maes = []
    best_val_mse=100000
    best_val_mae=100000

    for epoch in range(1,epoches+1):
        loss = train_epoch(model,training_input, training_target, 
                           x_train_mark,y_train_mark,
                           batch_size,
                           loss_criterion,optimizer,device)
        training_losses.append(loss)
        val_mse,val_mae = val_evaluate(model,val_input,val_target,
                                       x_test_mark,y_test_mark,
                                       means,stds,loss_criterion,device)
        validation_losses.append(val_mse)
        validation_maes.append(val_mae)
        
        if best_val_mse>val_mse:
            best_val_mse=val_mse
        if best_val_mae>val_mae:
            best_val_mae=val_mae
            torch.save(model,save_path)
        print(
            "Epoch {:05d} | Loss {:.4f} | val mse: {:.3f} (best {:.3f}) | val mae: {:.3f} (best {:.3f})".format(
                epoch,loss,val_mse,best_val_mse,val_mae,best_val_mae
            )
        )
    return training_losses,validation_losses,validation_maes

# In[]
def model_predict(model,x,data_stamp,
                  current_timestamp_str,stamp_scaler,
                  num_timesteps_input,num_timesteps_output,
                  means,stds,device="cuda"):
    x = torch.tensor(x[-num_timesteps_input:]).to(device)
    x = x.unsqueeze(dim=0).permute(0,2,1).unsqueeze(dim=-2).to(device)
    x_mark = data_stamp[-num_timesteps_input:].reshape(1,-1,5).to(device)
    y_mark = date_deal(new_step_mark(current_timestamp_str,
                                     steps=num_timesteps_output),
                       use_for_test=True,
                       stamp_scaler=stamp_scaler)[0].reshape(1,-1,5).to(device)
    
    with torch.no_grad():
        model.eval()
        out = model(x,x_mark,y_mark)
        out = out.detach().cpu().numpy()*stds+means
        return out,new_step_mark(current_timestamp_str,
                                 steps=num_timesteps_output)
# In[]
def new_step_mark(current_timestamp_str='2024-03-05 13:00',  
                  steps=4):  
    # 将字符串转换为datetime对象  
    current_time = datetime.datetime.strptime(current_timestamp_str, '%Y-%m-%d %H:%M')  
    # 设置股票交易时间列表（以小时和分钟为单位）  
    trade_times = [(10, 30), (11, 30), (14, 0), (15, 0)]  
    # 存储格式化后的日期时间字符串  
    formatted_times = []  
      
    # 当前日期  
    current_date = current_time.date()  
    # 当前交易日的剩余时间戳  
    remaining_steps_today = steps  
      
    # 生成当前交易日剩余的时间戳  
    for hour, minute in trade_times:  
        trade_time = datetime.datetime.combine(current_date, datetime.time(hour, minute))  
        if trade_time > current_time:  
            formatted_times.append(trade_time.strftime('%Y-%m-%d %H:%M'))  
            remaining_steps_today -= 1  
            if remaining_steps_today == 0:  
                break  
      
    # 如果步骤未填满，则继续到下一个交易日  
    while remaining_steps_today > 0:  
        current_date += datetime.timedelta(days=1)  # 下一个交易日  
        for hour, minute in trade_times:  
            trade_time = datetime.datetime.combine(current_date, datetime.time(hour, minute))  
            formatted_times.append(trade_time.strftime('%Y-%m-%d %H:%M'))  
            remaining_steps_today -= 1  
            if remaining_steps_today == 0:  
                break  
      
    # 将格式化后的日期时间字符串转换为Pandas Series对象  
    date_array = pd.Series(formatted_times)  
    return date_array  
# In[]
def former_predict(choose_channels,read_path,configs,
                    model_name,
                    num_timesteps_input=6,num_timesteps_output=1,
                    epoches=1000,device="cuda",
                    lr=1e-3,weight_decay=5e-4,batch_size=32,
                    train_size=0.6,val_size=0.2):
    tab_cols,x,date = time_data_load(read_path)
    x,tab_name = choose_col_analysis(x,tab_cols,choose_channels)
    data_stamp,stamp_scaler = date_deal(date)
    x,means,stds=normalization(x)
    means = means.reshape(1,-1,1,1)
    stds = stds.reshape(1,-1,1,1)
    # split_line1 = int(x.shape[0] * train_size)
    # split_line2 = int(x.shape[0] * (train_size+val_size))
    train_original_data = x #x[:split_line1]
    val_original_data = x #x[split_line1:split_line2]

    train_data_stamp = data_stamp #[:split_line1]
    val_data_stamp = data_stamp #[split_line1:split_line2]

    x_train,y_train,x_train_mark,y_train_mark = generate_dataset(train_original_data,
                                                data_stamp=train_data_stamp,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    x_val,y_val,x_val_mark,y_val_mark = generate_dataset(val_original_data,
                                                data_stamp=val_data_stamp,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    """
    模型基础设置
    """
    if model_name =='tcaffn':
        model = Model(configs).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_criterion = nn.L1Loss()
    training_losses,validation_losses,validation_maes=train(model,epoches,
                                    x_train,y_train,
                                    x_train_mark,y_train_mark,
                                    x_val,y_val,
                                    x_val_mark,y_val_mark,
                                    batch_size,
                                    loss_criterion,optimizer,
                                    means,stds,device,save_path=configs["save_path"])
    """
    测试集测试
    """
    model=torch.load(configs["save_path"])

    """
    训练情况
    """
    fig, ax = plt.subplots(figsize=(7,3),dpi=600)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(training_losses, label="train")
    plt.plot(validation_losses, label="valid")
    plt.title("Training for "+str([tab_name[i] for i in range(len(tab_name))]))
    plt.legend()
    plt.show()

    """
    mae验证情况
    """
    fig, ax = plt.subplots(figsize=(7,3),dpi=600)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(validation_maes, label="valid")
    plt.title("MAE validation for "+str([tab_name[i] for i in range(len(tab_name))]))
    plt.legend()
    plt.show()
    return model,means,stds,x,stamp_scaler,data_stamp
# In[]
if __name__ == '__main__':
    with open('choose_model.yaml','rb') as file:
        model_n = yaml.safe_load(file) 
    with open("model"+"\\"+model_n['name']+'.yaml','rb') as file:
        hyperparameters = yaml.safe_load(file) 

    model,means,stds,x,stamp_scaler,data_stamp = former_predict(
                     choose_channels=hyperparameters["choose_channels"],
                     read_path=hyperparameters["read_path"],
                     configs=hyperparameters,
                     model_name=model_n['name'],
                     num_timesteps_input=hyperparameters["inp_len"],
                     num_timesteps_output=hyperparameters["pred_len"],
                     epoches=hyperparameters["epoches"],
                     device=hyperparameters["device"],
                     lr=hyperparameters["lr"],
                     weight_decay=hyperparameters["weight_decay"],
                     batch_size=hyperparameters["batch_size"],
                     train_size=hyperparameters["train_size"],
                     val_size=hyperparameters["val_size"])
    
    ypred,date_list = model_predict(model,x,data_stamp,
                          current_timestamp_str=hyperparameters["current_timestamp_str"],
                          stamp_scaler=stamp_scaler,
                          num_timesteps_input=hyperparameters["inp_len"],
                          num_timesteps_output=hyperparameters["pred_len"],
                          means=means,
                          stds=stds,device="cuda")
    np.save("pred_save\ypred.npy",ypred)
    print("output shape:",ypred.shape)
    for i in range(ypred.shape[3]):
        print("day "+str(i+1)+":",ypred[0,:,:,i])
    ypred = ypred.squeeze().T
    pd.DataFrame(ypred,
                 columns=['close', 'high', 'low'],
                 index = date_list.tolist()).to_excel(hyperparameters["excel_path"])
    delete_file(hyperparameters["save_path"])
    delete_file(hyperparameters["read_path"])
# %%
