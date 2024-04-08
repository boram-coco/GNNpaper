import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import networkx as nx
import sklearn

# sklearn
import sklearn
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

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from sklearn.model_selection import train_test_split
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from sklearn.model_selection import train_test_split


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

def build_graph_tripartite(df_input, graph_type=nx.Graph()):
    df=df_input.copy()
    mapping={x:node_id for node_id, x in enumerate(set(df.index.values.tolist() + 
                                                       df["cc_num"].values.tolist() +
                                                       df["merchant"].values.tolist()))}
    df["in_node"]= df["cc_num"].apply(lambda x: mapping[x])
    df["out_node"]=df["merchant"].apply(lambda x:mapping[x])
    
        
    G=nx.from_edgelist([(x["in_node"], mapping[idx]) for idx, x in df.iterrows()] +\
                        [(x["out_node"], mapping[idx]) for idx, x in df.iterrows()], create_using=graph_type)
    
    nx.set_edge_attributes(G,{(x["in_node"], mapping[idx]):x["is_fraud"] for idx, x in df.iterrows()}, "label")
     
    nx.set_edge_attributes(G,{(x["out_node"], mapping[idx]):x["is_fraud"] for idx, x in df.iterrows()}, "label")
    
    nx.set_edge_attributes(G,{(x["in_node"], mapping[idx]):x["amt"] for idx, x in df.iterrows()}, "weight")
    
    nx.set_edge_attributes(G,{(x["out_node"], mapping[idx]):x["amt"] for idx, x in df.iterrows()}, "weight")
    
    
    return G
    


def try_book(fraudTrain, fraudrate, n, prev_results=None):
    if prev_results is None:
        df_results = pd.DataFrame(columns=[
            'model', 'time', 'acc', 'pre', 'rec', 'f1', 'auc', 'graph_based', 
            'method', 'throw_rate', 'train_size', 'train_cols', 'train_frate', 
            'test_size', 'test_frate', 'hyper_params'
        ])
    else:
        df_results = prev_results
    
    dfrate = throw(fraudTrain, fraudrate)
    df_tr, df_tst = sklearn.model_selection.train_test_split(dfrate)
        
    dfn = fraudTrain[::n]
    dfnn = dfn[~dfn.index.isin(df_tr.index)]
    dfnn = dfnn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfnn)
   
    df2, mask = concat(df_tr, df_tstn)
    df2['index'] = df2.index
    df = df2.reset_index()

    G_df = build_graph_tripartite(df, nx.Graph())

    train_edges, test_edges, train_labels, y = train_test_split(list(range(len(G_df.edges))), 
                                                                 list(nx.get_edge_attributes(G_df, "label").values()), 
                                                                 test_size=0.20, 
                                                                 random_state=42)

    edgs = list(G_df.edges)
    train_graph = G_df.edge_subgraph([edgs[x] for x in train_edges]).copy()
    train_graph.add_nodes_from(list(set(G_df.nodes) - set(train_graph.nodes)))

    node2vec_train = Node2Vec(train_graph, weight_key='weight')
    model_train = node2vec_train.fit(window=10)

    
    classes = [HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder]
    evaluation_results = {}
    
    for cl in classes:
        embeddings_train = cl(keyed_vectors=model_train.wv)

        train_embeddings = [embeddings_train[str(edgs[x][0]), str(edgs[x][1])] for x in train_edges]
        test_embeddings = [embeddings_train[str(edgs[x][0]), str(edgs[x][1])] for x in test_edges]

        rf = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf.fit(train_embeddings, train_labels)
    
        yhat = rf.predict(test_embeddings)
        acc = metrics.accuracy_score(y, yhat)
        pre = metrics.precision_score(y, yhat)
        rec = metrics.recall_score(y, yhat)
        f1 = metrics.f1_score(y, yhat)
        auc = metrics.roc_auc_score(y, yhat)

        c1 = cl.__name__
        result = {
            'model': 'bipartite',
            'time': None,
            'acc': acc,
            'pre': pre,
            'rec': rec,
            'f1': f1,
            'auc': auc,
            'graph_based': True,
            'method': 'Proposed',
            'throw_rate': df.is_fraud.mean(),
            'train_size': len(train_labels),
            'train_cols': 'amt',
            'train_frate': train_labels.is_fraud.mean(),
            'test_size': len(y),
            'test_frate': y.is_fraud.mean(),
            'hyper_params': None,
            'theta': None,
            'gamma': None
        }
        
        evaluation_results[c1] = result
    
    df_results = df_results.append(evaluation_results, ignore_index=True)
    
    return df_results