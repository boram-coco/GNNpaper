import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import networkx as nx
import sklearn
import xgboost as xgb

# sklearn
from sklearn import model_selection # split함수이용
from sklearn import ensemble # RF,GBM
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# gnn
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv


def throw(df, fraud_rate):  # 사기 거래 비율에 맞춰 버려지는 함수!
    df1 = df[df['is_fraud'] == 1].copy()
    df0 = df[df['is_fraud'] == 0].copy()
    df0_downsample = (len(df1) * (1-fraud_rate)) / (len(df0) * fraud_rate)
    df0_down = df0.sample(frac=df0_downsample, random_state=42)
    df_p = pd.concat([df1, df0_down])
    return df_p

def split_dataframe(data_frame, test_fraud_rate, test_rate=0.3):
    n = len(data_frame)

    # 사기 거래와 정상 거래를 분리
    fraud_data = data_frame[data_frame['is_fraud'] == 1]
    normal_data = data_frame[data_frame['is_fraud'] == 0]

    # 테스트 데이터 크기 계산
    test_samples = int(test_fraud_rate * (n * test_rate))
    remaining_test_samples = int(n * test_rate) - test_samples

    # 사기 거래 및 정상 거래에서 무작위로 테스트 데이터 추출
    test_fraud_data = fraud_data.sample(n=test_samples, replace=False)
    test_normal_data = normal_data.sample(n=remaining_test_samples, replace=False)

    # 테스트 데이터 합치기
    test_data = pd.concat([test_normal_data, test_fraud_data])

    # 훈련 데이터 생성
    train_data = data_frame[~data_frame.index.isin(test_data.index)]

    return train_data, test_data

def concat(df_tr, df_tst):   
    df = pd.concat([df_tr, df_tst])
    train_mask = np.concatenate((np.full(len(df_tr), True), np.full(len(df_tst), False)))    # index꼬이는거 방지하기 위해서? ★ (이거,, 훔,,?(
    test_mask =  np.concatenate((np.full(len(df_tr), False), np.full(len(df_tst), True))) 
    mask = (train_mask, test_mask)
    return df, mask

def evaluation(y, yhat):
    metrics = [sklearn.metrics.accuracy_score,
               sklearn.metrics.precision_score,
               sklearn.metrics.recall_score,
               sklearn.metrics.f1_score,
               sklearn.metrics.roc_auc_score]
    return pd.DataFrame({m.__name__:[m(y,yhat).round(6)] for m in metrics})

def compute_time_difference(group):
    n = len(group)
    result = []
    for i in range(n):
        for j in range(n):
            time_difference = abs((group.iloc[i].trans_date_trans_time - group.iloc[j].trans_date_trans_time).total_seconds())
            result.append([group.iloc[i].name, group.iloc[j].name, time_difference])
    return result

def edge_index(df, unique_col, theta, gamma):
    groups = df.groupby(unique_col)
    edge_index = np.array([item for sublist in (compute_time_difference(group) for _, group in groups) for item in sublist])
    edge_index = edge_index.astype(np.float64)
    # filename = f"edge_index{str(unique_col).replace(' ', '').replace('_', '')}.npy"  # 저장
    # np.save(filename, edge_index)
    edge_index[:,2] = (np.exp(-edge_index[:,2]/(theta)) != 1)*(np.exp(-edge_index[:,2]/(theta))).tolist()
    edge_index = torch.tensor([(int(row[0]), int(row[1])) for row in edge_index if row[2] > gamma], dtype=torch.long).t()
    return edge_index



def gcn_data(df):
    x = torch.tensor(df['amt'].values, dtype=torch.float).reshape(-1,1)
    y = torch.tensor(df['is_fraud'].values,dtype=torch.int64)
    data = torch_geometric.data.Data(x=x, edge_index = edge_index, y=y, train_mask = mask[0], test_mask= mask[1])
    return data


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
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    pred = model(data).argmax(dim=1)
    yyhat = pred[data.test_mask]
    
    return yyhat

# # 모델과 옵티마이저 생성
# model = GCN1()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# # 함수 호출
# yyhat = train_and_evaluate_model(data, model, optimizer)


def proposed_df(fraudTrain, fraud_rate, test_fraud_rate):
    df = throw(fraudTrain, fraud_rate)
    df_tr, df_tst = split_dataframe(df, test_fraud_rate)
    df2, mask = concat(df_Tr, df_tst)
    df2['index'] = df2.index
    df3 = df2.reset_index()
    return df3


def proposed_gcn(df):
    edge_index = edge_index(df, 'cc_num', theta, gamma)
    data = gcn_data(df)
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat = train_and_evaluate_model(data, model, optimizer)
    results1=evaluation(yy,yyhat)
    
    
def try_1(df, fraud_rate, test_fraud_rate, theta, gamma):
    df = proposed_df(df, fraud_rate, test_fraud_rate)
    edge_index = edge_index(df, 'cc_num', theta, gamma)
    data = gcn_data(df)
    model = GCN1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    yy = (data.y[data.test_mask]).numpy()
    yyhat = train_and_evaluate_model(data, model, optimizer)
    
    model = []
    time_diff = []
    acc = []
    pre = []
    rec = []
    f1 = [] 
    auc = [] 
    graph_based = []
    method = [] 
    train_size = []
    train_cols = []
    train_frate = []
    test_size = []
    test_frate = []
    hyper_params = [] 
    theta = []
    gamma = []
    
        
    for result in results:
        model_name = result['model']
        model.append('GCN')
        time_diff.append(None)  
        acc.append(result['acc']) 
        pre.append(result['pre'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        auc.append(result['auc'])
        graph_based.append(True) 
        method.append('Proposed') 
        train_size.append(len(tr))
        train_cols.append([col for col in tr.columns if col != 'is_fraud'])
        train_frate.append(tr.is_fraud.mean())
        test_size.append(len(tst))
        test_frate.append(tst.is_fraud.mean())
        hyper_params.append(None)
        
    df_results = pd.DataFrame(dict(
        model=model,
        time=time_diff,
        acc=acc,
        pre=pre,
        rec=rec,
        f1=f1,
        auc=auc,
        graph_based=graph_based,
        method=method,
        throw_rate=fraud_rate,  
        train_size=train_size,
        train_cols=train_cols,
        train_frate=train_frate,
        test_size=test_size,
        test_frate=test_frate,
        hyper_params=hyper_params,
        theta = theta,
        gamma = gamma
    ))    
    return df_results
    