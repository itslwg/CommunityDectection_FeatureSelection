import community

import pandas as pd
import numpy as np
import networkx as nx
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from utils.metrics import utility_score
from utils.cv import PurgedGroupTimeSeriesSplit
from utils.plots import plot_cv_indices


def clean_matrix(C, T, N):
    """Random Matrix Theory correlation matrix cleaning

    Args:
        C (np.array): Correlation matrix
        T (int): Number of rows
        N (int): Number of columns

    Returns:
        np.array: Cleaned correlation matrix
    """
    lambda_plus = (1+np.sqrt(N/T))**2
    values, vectors = np.linalg.eigh(C)
    vectors = np.matrix(vectors)  
    
    C_clean = np.zeros(shape=(N,N))
    
    for i in range(N-1):
        if values[i]>lambda_plus:
            C_clean = C_clean + values[i]* np.dot(vectors[i,].T,vectors[i,])    
    return C_clean


def clean_matrix_(X_tr_norm):
    """Helper to clean_matrix."""    
    C_tr = np.corrcoef(X_tr_norm, rowvar=False)  # Faster than pd.DataFrame.corr()
    T, N = X_tr_norm.shape
    return clean_matrix(C_tr, T, N)


def mda_clustered(clf, df, X, y, clusters, n_splits=5, cv = None, seed=42):
    """Clustered MDA.
    
    Source: 
        López de Prado, M. (2020). 
        Machine Learning for Asset Managers (Elements in Quantitative Finance).
        Cambridge: Cambridge University Press. doi:10.1017/9781108883658
    """
    np.random.seed(42)
    if not cv:
        # Train on max 3 months, test on 1 month max
        cv = PurgedGroupTimeSeriesSplit(
            n_splits = n_splits,
            max_train_group_size=63,
            max_test_group_size=21,
            group_gap=10
        )
    scr0, scr1 = pd.Series(), pd.DataFrame(columns = clusters.keys())
    X = X.reset_index(drop=True)
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for i, (train,test) in enumerate(tqdm(cv.split(X=X, y=y, groups=df['date']))):
        
        X0, y0 = X.iloc[train,:], y.iloc[train]
        X1, y1 = X.iloc[test,:], y.iloc[test]
        df_test = df.iloc[test, :]

        fit = clf.fit(X = X0, y = y0)
        y_pred = fit.predict(X = X1)
        scr0.loc[i] = utility_score(
            df_test['date'].values, 
            df_test['weight'].values, 
            df_test['resp'].values, 
            y_pred
        )
        #prob = fit.predict_proba(X1)
        # scr0.loc[i] = -log_loss(y1, prob, labels = clf.classes_)
        for j in scr1.columns:
            X1_ = X1.copy(deep=True)
            df_test_ = df_test.copy(deep=True)
            for k in clusters[j]:
                np.random.shuffle(X1_[k].values) # shuffle cluster
                y_pred = fit.predict(X1_)
                scr1.loc[i,j] = utility_score(
                    df_test_['date'].values, 
                    df_test_['weight'].values, 
                    df_test_['resp'].values, 
                    y_pred
                )
                # prob = fit.predict_proba(X1_)
                # scr1.loc[i,j] = -log_loss(y1, prob, labels = clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    #imp = imp / (-1 * scr1)
    imp = pd.concat({
        "mean": np.mean(np.abs(imp)),
        "std": imp.std() * imp.shape[0] ** -0.5
    }, axis = 1)
    
    return imp.reset_index()


def louvain_clustering(C, random_state=None):
    """Louvain clustering pipeline
    
    Select best partition and create graph.
    Source: 
        Damien Challet, FBD Lecture notes
    """
    C_clean = C.copy()
    C_clean = np.abs(C_clean)
    graph = nx.from_numpy_matrix(C_clean)
    part = community.community_louvain.best_partition(graph, random_state = random_state)
    
    res_df = pd.DataFrame.from_dict(part,orient="index", columns=["cluster"])
    res_df.sort_index(inplace=True)
    
    return res_df, graph


def runN_Louvain(C, N=100, verbose=True, random_state=None):
    """Run Louvain clustering multiple times for robustness

    Args:
        C (np.array): Correlation matrix
        N (int, optional): Number of runs. Defaults to 100.

    Returns:
        pd.DataFrame: Partitions
        nx.Graph: Graph
    """
    graph = nx.from_numpy_matrix(np.abs(C))
    clus_res = []
    for _ in (tqdm(range(N)) if verbose else range(N)):
        part, _ = louvain_clustering(C, random_state=random_state)
        part["feature"] = part.index
        clus_res.append(part)
    clus_res = pd.concat(clus_res, ignore_index=True)
    count_feat_in_clus = clus_res.groupby(["feature","cluster"]).size()
    upt_part = count_feat_in_clus.groupby(level=[0]).idxmax()
    upt_part = pd.DataFrame(upt_part.tolist(), columns=["xx","cluster"])["cluster"].to_frame()
    return upt_part, graph


def select_k_features(partitions, ns):
    """Subset features based on cluster importance."""
    def select_features(s, nm, ns):
        n = ns.loc[nm]
        s_ = s.sort_values("importance", ascending=False)
        return s.iloc[:n].drop("cluster", axis=1)
    
    features = partitions.groupby("cluster").apply(
        lambda g: select_features(s=g, nm=g.name, ns=ns)
    ).reset_index(level=1, drop=True)

    return list(features.feature)