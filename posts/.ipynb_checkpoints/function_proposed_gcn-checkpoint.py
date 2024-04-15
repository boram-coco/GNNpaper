import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import networkx as nx
import sklearn

# sklearn
from sklearn import model_selection # split함수이용
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
# gnn
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
import time
import datetime


def throw(df, fraud_rate, random_state=42):  # 사기 거래 비율에 맞춰 버려지는 함수!
    df1 = df[df['is_fraud'] == 1].copy()
    df0 = df[df['is_fraud'] == 0].copy()
    df0_downsample = (len(df1) * (1-fraud_rate)) / (len(df0) * fraud_rate)
    df0_down = df0.sample(frac=df0_downsample, random_state=random_state)
    df_p = pd.concat([df1, df0_down])
    return df_p

def split_dataframe(data_frame, test_fraud_rate, test_rate=0.3, random_state=42):
    n = len(data_frame)

    # 사기 거래와 정상 거래를 분리
    fraud_data = data_frame[data_frame['is_fraud'] == 1]
    normal_data = data_frame[data_frame['is_fraud'] == 0]

    # 테스트 데이터 크기 계산
    test_samples = int(test_fraud_rate * (n * test_rate))
    remaining_test_samples = int(n * test_rate) - test_samples

    # 사기 거래 및 정상 거래에서 무작위로 테스트 데이터 추출
    test_fraud_data = fraud_data.sample(n=test_samples, replace=False, random_state=random_state)
    test_normal_data = normal_data.sample(n=remaining_test_samples, replace=False, random_state=random_state)

    # 테스트 데이터 합치기
    test_data = pd.concat([test_normal_data, test_fraud_data])

    # 훈련 데이터 생성
    train_data = data_frame[~data_frame.index.isin(test_data.index)]

    return train_data, test_data


####################################################################################################################
################ proposed 

def concat(df_tr, df_tst):   
    df = pd.concat([df_tr, df_tst])
    train_mask = np.concatenate((np.full(len(df_tr), True), np.full(len(df_tst), False)))    # index꼬이는거 방지하기 위해서? ★ (이거,, 훔,,?(
    test_mask =  np.concatenate((np.full(len(df_tr), False), np.full(len(df_tst), True))) 
    mask = (train_mask, test_mask)
    return df, mask
    
def evaluate(y, yhat):
    y = pd.Series(np.array(y).reshape(-1))
    yhat = pd.Series(np.array(yhat).reshape(-1))
    nan_rate = pd.Series(yhat).isna().mean()
    nan_index = pd.Series(yhat).isna()
    y = np.array(y)[~nan_index]
    yhat_prob = np.array(yhat)[~nan_index]
    yhat_01 = yhat_prob > 0.5
    acc = sklearn.metrics.accuracy_score(y,yhat_01)
    pre = sklearn.metrics.precision_score(y,yhat_01)
    rec = sklearn.metrics.recall_score(y,yhat_01)
    f1 = sklearn.metrics.f1_score(y,yhat_01)
    auc = sklearn.metrics.roc_auc_score(y,yhat_prob)
    return {'acc':acc,'pre':pre,'rec':rec,'f1':f1,'auc':auc}
    
def compute_time_difference(group):
    n = len(group)
    result = []
    for i in range(n):
        for j in range(n):
            time_difference = abs((group.iloc[i].trans_date_trans_time - group.iloc[j].trans_date_trans_time).total_seconds())
            result.append([group.iloc[i].name, group.iloc[j].name, time_difference])
    return result

# def edge_index(df, theta, gamma):
#     groups = df.groupby('cc_num')
#     edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
#     edge_index = edge_index.astype(np.float64)
#     # filename = f"edge_index{str(unique_col).replace(' ', '').replace('_', '')}.npy"  # 저장
#     # np.save(filename, edge_index)
#     edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
#     edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
#     return edge_index

# def gcn_data(df):
#     x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
#     y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
#     data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
#     return data


class GCN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

def train_and_evaluate_model(data, model, optimizer, num_epochs=400):
    t1 = time.time()
    model.train()
    t2 = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    pred = model(data).argmax(dim=1)
    yyhat = pred[data.test_mask]
    yhat_ = model(data)[:,-1][data.test_mask] 
    return yyhat, yhat_

# # 모델과 옵티마이저 생성
# model = GCN1()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# # 함수 호출
# yyhat = train_and_evaluate_model(data, model, optimizer)


def evaluation(y, yhat, yhat_):
    acc = sklearn.metrics.accuracy_score(y,yhat)
    pre = sklearn.metrics.precision_score(y,yhat)
    rec = sklearn.metrics.recall_score(y,yhat)
    f1 = sklearn.metrics.f1_score(y,yhat)
    auc = sklearn.metrics.roc_auc_score(y,yhat_)
    return {'acc':acc,'pre':pre,'rec':rec,'f1':f1,'auc':auc}
    



def proposed_df(fraudTrain, fraud_rate, test_fraud_rate):
    df = throw(fraudTrain, fraud_rate)
    df_tr, df_tst = split_dataframe(df, test_fraud_rate)
    df2, mask = concat(df_tr, df_tst)
    df2['index'] = df2.index
    df3 = df2.reset_index()
    return df3, df_tr, df_tst, mask


import pandas as pd
import numpy as np
import torch
import torch_geometric
from function_proposed_gcn import proposed_df, compute_time_difference, GCN1, train_and_evaluate_model

def try_1(df, fraud_rate, test_fraud_rate, theta, gamma, prev_results=None):
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results.copy()
    
        df = throw(df, fraud_rate)
        df_tr, df_tst = split_dataframe(df, test_fraud_rate) 
    
        df, mask = concat(df_tr, df_tst)
    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tst),
        'test_frate': df_tst.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results


def try_11(fraudTrain, fraud_rate, theta, gamma, prev_results=None):
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results.copy()
        
    df = throw(fraudTrain, fraud_rate)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df)
    
    df2, mask = concat(df_tr, df_tst)
    df2['index'] = df2.index
    df3 = df2.reset_index()
    
    groups = df3.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df3['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df3['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df3.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tst),
        'test_frate': df_tst.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results


def proposed_df2(df_tr, df_tst):
    df2, mask = concat(df_tr, df_tst)
    df2['index'] = df2.index
    return df2, df_tr, df_tst, mask

def try_2(df, n,  fraud_rate, test_fraud_rate, theta, gamma, prev_results=None):
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results.copy()
        
    df = df[::n]
    df = df.reset_index(drop=True)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df)
    df = pd.concat([df_tr, df_tst])
    train_mask = np.concatenate((np.full(len(df_tr), True), np.full(len(df_tst), False)))    # index꼬이는거 방지하기 위해서? ★ (이거,, 훔,,?(
    test_mask =  np.concatenate((np.full(len(df_tr), False), np.full(len(df_tst), True))) 
    mask = (train_mask, test_mask)

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': fraud_rate,
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tst),
        'test_frate': test_fraud_rate,
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results


def try_3(df, n, fraud_rate, test_fraud_rate, theta, gamma, prev_results=None): # 인덱스 겹침
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results
    
    df50 = throw(df,0.5)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
        
    dfn = df[::n]
    dfn = dfn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfn)
    

   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': fraud_rate,
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tst),
        'test_frate': test_fraud_rate,
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results

def try_4(fraudTrain, n, theta, gamma, prev_results=None):  # 인덱스 겹침
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results
    
    df50 = throw(fraudTrain,0.5)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
        
    dfn = fraudTrain[::n]
    dfn = dfn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfn)
    

   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tstn),
        'test_frate':  df_tstn.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results


def try_5(fraudTrain, n, theta, gamma, prev_results=None): # try_4에서 index안겹치게 바꾼것
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results
    
    df50 = throw(fraudTrain,0.5)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
        
    dfn = fraudTrain[::n]
    dfnn = dfn[~dfn.index.isin(df_tr.index)]
    dfnn = dfnn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfnn)
    

   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tstn),
        'test_frate':  df_tstn.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results



def try_6(fraudTrain, fraudrate, theta, gamma, prev_results=None): # 이게 그냥 tr/tst비율 바로 맞출수 있지 않을까?
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results
    
    df = throw(fraudTrain,fraudrate)
    df_tr, df_tstn = sklearn.model_selection.train_test_split(df)
    
   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tstn),
        'test_frate':  df_tstn.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results

def try_7(fraudTrain, ratio, n, theta, gamma, prev_results=None): # try_4에서 index안겹치게 바꾼것
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params', 'theta', 'gamma'
        ])
    else:
        df_results = prev_results
    
    df50 = throw(fraudTrain,ratio)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
        
    dfn = fraudTrain[::n]
    dfnn = dfn[~dfn.index.isin(df_tr.index)]
    dfnn = dfnn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfnn)
    

   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    
    groups = df.groupby('cc_num')
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat, yyhat_ = train_and_evaluate_model(data, model, optimizer)
    yyhat_ = yyhat_.detach().numpy()
    eval = evaluation(yy, yyhat, yyhat_)
    
    result = {
        'model': 'GCN',
        'time': None,
        'acc': eval['acc'],
        'pre': eval['pre'],
        'rec': eval['rec'],
        'f1': eval['f1'],
        'auc': eval['auc'],
        'graph_based': True,
        'method': 'Proposed',
        'throw_rate': df.is_fraud.mean(),
        'train_size': len(df_tr),
        'train_cols': 'amt',
        'train_frate': df_tr.is_fraud.mean(),
        'test_size': len(df_tstn),
        'test_frate':  df_tstn.is_fraud.mean(),
        'hyper_params': None,
        'theta': theta,
        'gamma': gamma
    }
    
    df_results = df_results.append(result, ignore_index=True)
    
    return df_results

