import pandas as pd
import numpy as np
import sklearn
import pickle 
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



####################################################################################################################
################ pyod


from pyod.models.abod import ABOD
#from pyod.models.alad import ALAD
#from pyod.models.anogan import AnoGAN
#from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.cd import CD
from pyod.models.copod import COPOD
#from pyod.models.deep_svdd import DeepSVDD
#from pyod.models.dif import DIF
from pyod.models.ecod import ECOD
#from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.kpca import PyODKernelPCA
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
#from pyod.models.lunar import LUNAR
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
#from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.qmcd import QMCD
from pyod.models.rgraph import RGraph
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
#from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
#from pyod.models.suod import SUOD
#from pyod.models.vae import VAE
#from pyod.models.xgbod import XGBOD

def pyod_preprocess(fraudTrain, n):
    df = fraudTrain[::n]
    df = df.reset_index()
    df_tr, df_tst = sklearn.model_selection.train_test_split(df)
    df = pd.concat([df_tr, df_tst])
     
    X = pd.DataFrame(df_tr['amt'])
    y = pd.DataFrame(df_tr['is_fraud'])
    XX = pd.DataFrame(df_tst['amt'])
    yy = pd.DataFrame(df_tst['is_fraud'])
    throw_rate = df.is_fraud.mean()
    fraud_ratio = df_tr.is_fraud.mean()
    predictors = {
    'ABOD': ABOD(contamination=fraud_ratio),
#    'ALAD': ALAD(contamination=fraud_ratio),
#    'AnoGAN': AnoGAN(contamination=fraud_ratio),
#    'AutoEncoder':AutoEncoder(contamination=fraud_ratio),
##    'CBLOF': CBLOF(contamination=fraud_ratio,n_clusters=2),
##    'COF': COF(contamination=fraud_ratio),
##    'CD': CD(contamination=fraud_ratio),
    'COPOD': COPOD(contamination=fraud_ratio),
#    'DeepSVDD': DeepSVDD(contamination=fraud_ratio),
#    'DIF': DIF(contamination=fraud_ratio),    
    'ECOD': ECOD(contamination=fraud_ratio),
#    'FeatureBagging': FeatureBagging(contamination=fraud_ratio),
    'GMM': GMM(contamination=fraud_ratio),
    'HBOS': HBOS(contamination=fraud_ratio),
    'IForest': IForest(contamination=fraud_ratio),
    'INNE': INNE(contamination=fraud_ratio),
    'KDE': KDE(contamination=fraud_ratio),
    'KNN': KNN(contamination=fraud_ratio),
####    'KPCA': KPCA(contamination=fraud_ratio),
#    'PyODKernelPCA': PyODKernelPCA(contamination=fraud_ratio),
##    'LMDD': LMDD(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
    'LOF': LOF(contamination=fraud_ratio),
####    'LOCI': LOCI(contamination=fraud_ratio),
#    'LUNAR': LUNAR(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
#    'LSCP': LSCP(contamination=fraud_ratio),
    'MAD': MAD(contamination=fraud_ratio),
    'MCD': MCD(contamination=fraud_ratio),
#    'MO_GAAL': MO_GAAL(contamination=fraud_ratio),
    'OCSVM': OCSVM(contamination=fraud_ratio),
    'PCA': PCA(contamination=fraud_ratio),
###    'QMCD': QMCD(contamination=fraud_ratio),
####    'RGraph': RGraph(contamination=fraud_ratio),
    'ROD': ROD(contamination=fraud_ratio),
##    'Sampling': Sampling(contamination=fraud_ratio),
##   'SOD': SOD(contamination=fraud_ratio),
#    'SO_GAAL': SO_GAAL(contamination=fraud_ratio),
####    'SOS': SOS(contamination=fraud_ratio),
#    'SUOD': SUOD(contamination=fraud_ratio),
#    'VAE': VAE(contamination=fraud_ratio),
#    'XGBOD': XGBOD(contamination=fraud_ratio),  
}
    return X, XX, y, yy, predictors, throw_rate


def pyod_preprocess2(fraudTrain,n):
 
    df50 = throw(fraudTrain,0.5)
    df_tr, df_tstn = sklearn.model_selection.train_test_split(df50)
        
    dfn = fraudTrain[::n]
    dfn = dfn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfn)
    
    dff = pd.concat([df_tr, df_tstn])

     
    X = pd.DataFrame(df_tr['amt'])
    y = pd.DataFrame(df_tr['is_fraud'])
    XX = pd.DataFrame(df_tstn['amt'])
    yy = pd.DataFrame(df_tstn['is_fraud'])
    throw_rate = dff.is_fraud.mean()
    fraud_ratio = df_tr.is_fraud.mean()
    predictors = {
    'ABOD': ABOD(contamination=fraud_ratio),
#    'ALAD': ALAD(contamination=fraud_ratio),
#    'AnoGAN': AnoGAN(contamination=fraud_ratio),
#    'AutoEncoder':AutoEncoder(contamination=fraud_ratio),
##    'CBLOF': CBLOF(contamination=fraud_ratio,n_clusters=2),
##    'COF': COF(contamination=fraud_ratio),
##    'CD': CD(contamination=fraud_ratio),
    'COPOD': COPOD(contamination=fraud_ratio),
#    'DeepSVDD': DeepSVDD(contamination=fraud_ratio),
#    'DIF': DIF(contamination=fraud_ratio),    
    'ECOD': ECOD(contamination=fraud_ratio),
#    'FeatureBagging': FeatureBagging(contamination=fraud_ratio),
    'GMM': GMM(contamination=fraud_ratio),
    'HBOS': HBOS(contamination=fraud_ratio),
    'IForest': IForest(contamination=fraud_ratio),
    'INNE': INNE(contamination=fraud_ratio),
    'KDE': KDE(contamination=fraud_ratio),
    'KNN': KNN(contamination=fraud_ratio),
####    'KPCA': KPCA(contamination=fraud_ratio),
#    'PyODKernelPCA': PyODKernelPCA(contamination=fraud_ratio),
##    'LMDD': LMDD(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
    'LOF': LOF(contamination=fraud_ratio),
####    'LOCI': LOCI(contamination=fraud_ratio),
#    'LUNAR': LUNAR(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
#    'LSCP': LSCP(contamination=fraud_ratio),
    'MAD': MAD(contamination=fraud_ratio),
    'MCD': MCD(contamination=fraud_ratio),
#    'MO_GAAL': MO_GAAL(contamination=fraud_ratio),
    'OCSVM': OCSVM(contamination=fraud_ratio),
    'PCA': PCA(contamination=fraud_ratio),
###    'QMCD': QMCD(contamination=fraud_ratio),
####    'RGraph': RGraph(contamination=fraud_ratio),
    'ROD': ROD(contamination=fraud_ratio),
##    'Sampling': Sampling(contamination=fraud_ratio),
##   'SOD': SOD(contamination=fraud_ratio),
#    'SO_GAAL': SO_GAAL(contamination=fraud_ratio),
####    'SOS': SOS(contamination=fraud_ratio),
#    'SUOD': SUOD(contamination=fraud_ratio),
#    'VAE': VAE(contamination=fraud_ratio),
#    'XGBOD': XGBOD(contamination=fraud_ratio),  
}
    return X, XX, y, yy, predictors, throw_rate


def pyod_preprocess3(fraudTrain,n):
 
    df50 = throw(fraudTrain,0.45)
    df_tr, df_tstn = sklearn.model_selection.train_test_split(df50)
        
    dfn = fraudTrain[::n]
    dfnn = dfn[~dfn.index.isin(df_tr.index)]
    dfnn = dfnn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfnn)
    
    dff = pd.concat([df_tr, df_tstn])

     
    X = pd.DataFrame(df_tr['amt'])
    y = pd.DataFrame(df_tr['is_fraud'])
    XX = pd.DataFrame(df_tstn['amt'])
    yy = pd.DataFrame(df_tstn['is_fraud'])
    throw_rate = dff.is_fraud.mean()
    fraud_ratio = df_tr.is_fraud.mean()
    predictors = {
#######    'ABOD': ABOD(contamination=fraud_ratio),
#    'ALAD': ALAD(contamination=fraud_ratio),
#    'AnoGAN': AnoGAN(contamination=fraud_ratio),
#    'AutoEncoder':AutoEncoder(contamination=fraud_ratio),
##    'CBLOF': CBLOF(contamination=fraud_ratio,n_clusters=2),
##    'COF': COF(contamination=fraud_ratio),
##    'CD': CD(contamination=fraud_ratio),
#######    'COPOD': COPOD(contamination=fraud_ratio),
#    'DeepSVDD': DeepSVDD(contamination=fraud_ratio),
#    'DIF': DIF(contamination=fraud_ratio),    
    'ECOD': ECOD(contamination=fraud_ratio),
#    'FeatureBagging': FeatureBagging(contamination=fraud_ratio),
    'GMM': GMM(contamination=fraud_ratio),
    'HBOS': HBOS(contamination=fraud_ratio),
    'IForest': IForest(contamination=fraud_ratio),
    'INNE': INNE(contamination=fraud_ratio),
    'KDE': KDE(contamination=fraud_ratio),
    'KNN': KNN(contamination=fraud_ratio),
####    'KPCA': KPCA(contamination=fraud_ratio),
#    'PyODKernelPCA': PyODKernelPCA(contamination=fraud_ratio),
##    'LMDD': LMDD(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
    'LOF': LOF(contamination=fraud_ratio),
####    'LOCI': LOCI(contamination=fraud_ratio),
#    'LUNAR': LUNAR(contamination=fraud_ratio),
    'LODA': LODA(contamination=fraud_ratio),
#    'LSCP': LSCP(contamination=fraud_ratio),
    'MAD': MAD(contamination=fraud_ratio),
    'MCD': MCD(contamination=fraud_ratio),
#    'MO_GAAL': MO_GAAL(contamination=fraud_ratio),
    'OCSVM': OCSVM(contamination=fraud_ratio),
    'PCA': PCA(contamination=fraud_ratio),
###    'QMCD': QMCD(contamination=fraud_ratio),
####    'RGraph': RGraph(contamination=fraud_ratio),
    'ROD': ROD(contamination=fraud_ratio),
##    'Sampling': Sampling(contamination=fraud_ratio),
##   'SOD': SOD(contamination=fraud_ratio),
#    'SO_GAAL': SO_GAAL(contamination=fraud_ratio),
####    'SOS': SOS(contamination=fraud_ratio),
#    'SUOD': SUOD(contamination=fraud_ratio),
#    'VAE': VAE(contamination=fraud_ratio),
#    'XGBOD': XGBOD(contamination=fraud_ratio),  
}
    return X, XX, y, yy, predictors, throw_rate


def pyod(X,XX,y,yy,predictors,throw_rate):
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
    for name, predictor in predictors.items():
        t1 = time.time()
        predictor.fit(X,y)
        t2 = time.time()
        yyhat = predictor.predict(XX)
        scores = evaluate(yy,yyhat)
        model.append(name)
        time_diff.append(t2-t1)
        acc.append(scores['acc'])
        pre.append(scores['pre'])
        rec.append(scores['rec'])
        f1.append(scores['f1'])
        auc.append(scores['auc'])
        graph_based.append(False)
        method.append('pyod')
        train_size.append(len(y)),
        train_cols.append(list(X.columns)),
        train_frate.append(np.array(y).reshape(-1).mean()),
        test_size.append(len(yy)),
        test_frate.append(np.array(yy).reshape(-1).mean())
        hyper_params.append(None)
    df_results = pd.DataFrame(dict(
        model = model,
        time=time_diff,
        acc=acc,
        pre=pre,
        rec=rec,
        f1=f1,
        auc=auc,
        graph_based = graph_based,
        method = method,
        throw_rate = throw_rate,
        train_size = train_size,
        train_cols = train_cols,
        train_frate = train_frate,
        test_size = test_size,
        test_frate = test_frate,
        hyper_params = hyper_params
    ))
    ymdhms = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
    df_results.to_csv(f'../results/{ymdhms}-pyod.csv',index=False)
    return df_results




###