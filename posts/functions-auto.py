import pandas as pd
import numpy as np
import sklearn
import pickle 
import time
import datetime
# autogluon
from autogluon.tabular import TabularDataset, TabularPredictor

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


def auto_amt_nb(fraudTrain, n):
    df = fraudTrain[["amt","is_fraud"]]
    df = df[::n]
    df = df.reset_index()
    df_tr, df_tst = sklearn.model_selection.train_test_split(df)
    tr = TabularDataset(df_tr)
    tst = TabularDataset(df_tst)
    predictr = TabularPredictor(label="is_fraud", verbosity=0)
    t1 = time.time()
    predictr.fit(tr) #presets='best_quality'
    t2 = time.time()
    time_diff = t2 - t1
    models = predictr._trainer.model_graph.nodes
    results = []
    for model_name in models:
    # 모델 평가
        eval_result = predictr.evaluate(tst, model=model_name)

    # 결과를 데이터프레임에 추가
        results.append({'model': model_name, 
                        'acc': eval_result['accuracy'], 
                        'pre': eval_result['precision'], 
                        'rec': eval_result['recall'], 
                        'f1': eval_result['f1'], 
                        'auc': eval_result['roc_auc']})
        
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
    
    for result in results:
        model_name = result['model']
        model.append(model_name)
        time_diff.append(None)  # 각 모델별로 학습한 시간을 나타내고 싶은데 잘 안됨
        acc.append(result['acc']) 
        pre.append(result['pre'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        auc.append(result['auc'])
        graph_based.append(False) 
        method.append('Auto_not_best') 
        throw_rate = df.is_fraud.mean()
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
        throw_rate=throw_rate,  
        train_size=train_size,
        train_cols=train_cols,
        train_frate=train_frate,
        test_size=test_size,
        test_frate=test_frate,
        hyper_params=hyper_params
    ))    
    ymdhms = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
    df_results.to_csv(f'../results/{ymdhms}-Autogluon_nb.csv',index=False)
    return df_results

def auto_amt_nb2(fraudTrain, n):
    
    df = fraudTrain[["amt","is_fraud"]]   
    df50 = throw(df,0.5)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
        
    dfn = df[::n]
    dfn = dfn.reset_index(drop=True)
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfn)
    
    dff = pd.concat([df_tr, df_tstn])
     
    tr = TabularDataset(df_tr)
    tst = TabularDataset(df_tstn)
    predictr = TabularPredictor(label="is_fraud", verbosity=0)
    t1 = time.time()
    predictr.fit(tr) #presets='best_quality'
    t2 = time.time()
    time_diff = t2 - t1
    models = predictr._trainer.model_graph.nodes
    results = []
    for model_name in models:
    # 모델 평가
        eval_result = predictr.evaluate(tst, model=model_name)

    # 결과를 데이터프레임에 추가
        results.append({'model': model_name, 
                        'acc': eval_result['accuracy'], 
                        'pre': eval_result['precision'], 
                        'rec': eval_result['recall'], 
                        'f1': eval_result['f1'], 
                        'auc': eval_result['roc_auc']})
        
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
    
    for result in results:
        model_name = result['model']
        model.append(model_name)
        time_diff.append(None)  # 각 모델별로 학습한 시간을 나타내고 싶은데 잘 안됨
        acc.append(result['acc']) 
        pre.append(result['pre'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        auc.append(result['auc'])
        graph_based.append(False) 
        method.append('Auto_not_best') 
        throw_rate = dff.is_fraud.mean()
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
        throw_rate=throw_rate,  
        train_size=train_size,
        train_cols=train_cols,
        train_frate=train_frate,
        test_size=test_size,
        test_frate=test_frate,
        hyper_params=hyper_params
    ))    
    ymdhms = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
    df_results.to_csv(f'../results/{ymdhms}-Autogluon_nb.csv',index=False)
    return df_results

def auto_amt_nb3(fraudTrain, n):
    
    df = fraudTrain[["amt","is_fraud"]]   
    df50 = throw(df,0.5)
    df_tr, df_tst = sklearn.model_selection.train_test_split(df50)
    
    dfn = fraudTrain[::n]
    dfnn = dfn[~dfn.index.isin(df_tr.index)]
    dfnn = dfnn.reset_index(drop=True)
    
    df_trn, df_tstn = sklearn.model_selection.train_test_split(dfnn)
    
    dff = pd.concat([df_tr, df_tstn])
     
    tr = TabularDataset(df_tr)
    tst = TabularDataset(df_tstn)
    predictr = TabularPredictor(label="is_fraud", verbosity=0)
    t1 = time.time()
    predictr.fit(tr) #presets='best_quality'
    t2 = time.time()
    time_diff = t2 - t1
    models = predictr._trainer.model_graph.nodes
    results = []
    for model_name in models:
    # 모델 평가
        eval_result = predictr.evaluate(tst, model=model_name)

    # 결과를 데이터프레임에 추가
        results.append({'model': model_name, 
                        'acc': eval_result['accuracy'], 
                        'pre': eval_result['precision'], 
                        'rec': eval_result['recall'], 
                        'f1': eval_result['f1'], 
                        'auc': eval_result['roc_auc']})
        
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
    
    for result in results:
        model_name = result['model']
        model.append(model_name)
        time_diff.append(None)  # 각 모델별로 학습한 시간을 나타내고 싶은데 잘 안됨
        acc.append(result['acc']) 
        pre.append(result['pre'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        auc.append(result['auc'])
        graph_based.append(False) 
        method.append('Auto_not_best') 
        throw_rate = dff.is_fraud.mean()
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
        throw_rate=throw_rate,  
        train_size=train_size,
        train_cols=train_cols,
        train_frate=train_frate,
        test_size=test_size,
        test_frate=test_frate,
        hyper_params=hyper_params
    ))    
    ymdhms = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
    df_results.to_csv(f'../results/{ymdhms}-Autogluon_nb.csv',index=False)
    return df_results


def auto_amt_nb4(fraudTrain, fraudrate):
    
    
    df = fraudTrain[["amt","is_fraud"]]   
    
      
    df = throw(df,fraudrate)
    df_tr, df_tstn = sklearn.model_selection.train_test_split(df)
    
   
     
    tr = TabularDataset(df_tr)
    tst = TabularDataset(df_tstn)
    predictr = TabularPredictor(label="is_fraud", verbosity=0)
    t1 = time.time()
    predictr.fit(tr) #presets='best_quality'
    t2 = time.time()
    time_diff = t2 - t1
    models = predictr._trainer.model_graph.nodes
    results = []
    for model_name in models:
    # 모델 평가
        eval_result = predictr.evaluate(tst, model=model_name)

    # 결과를 데이터프레임에 추가
        results.append({'model': model_name, 
                        'acc': eval_result['accuracy'], 
                        'pre': eval_result['precision'], 
                        'rec': eval_result['recall'], 
                        'f1': eval_result['f1'], 
                        'auc': eval_result['roc_auc']})
        
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
    
    for result in results:
        model_name = result['model']
        model.append(model_name)
        time_diff.append(None)  # 각 모델별로 학습한 시간을 나타내고 싶은데 잘 안됨
        acc.append(result['acc']) 
        pre.append(result['pre'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        auc.append(result['auc'])
        graph_based.append(False) 
        method.append('Auto_not_best') 
        throw_rate = df.is_fraud.mean()
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
        throw_rate=throw_rate,  
        train_size=train_size,
        train_cols=train_cols,
        train_frate=train_frate,
        test_size=test_size,
        test_frate=test_frate,
        hyper_params=hyper_params
    ))    
    ymdhms = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
    df_results.to_csv(f'../results/{ymdhms}-Autogluon_nb.csv',index=False)
    return df_results