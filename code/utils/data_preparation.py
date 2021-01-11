import sparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from utils.clustering import *
from utils.plots import *

from lightgbm import LGBMClassifier


def retransform(arr: np.array, df: pd.DataFrame) -> pd.DataFrame:
    """Helper for scikit learn preprocessing."""
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def impute(X, strategy, imputer=None):
    """Missing value imputer."""
    if imputer:
        X = retransform(imputer.transform(X), X)
    else:
        imputer = SimpleImputer(strategy=strategy)
        X = retransform(imputer.fit_transform(X), X)

    return X, imputer


def cwfs(K, partitions,
         df, X_tr, y_tr, 
         tr, cv = None,
         plot_importance = False):
    """Cluster Weighted Feature Selection."""
    sub_features = None
    plt = None
    ns = None
    if K != 'n_clusters':
        ## Determine the cluster importance using MDA.
        p = partitions.reset_index()
        p["index"] = ["feature_" + str(s) for s in p["index"].copy()]
        cs = p.groupby("cluster")["index"].apply(list)
        clusters = cs.to_dict()
    
        # Determine cluster importance
        imp_c = mda_clustered(
            LogisticRegression(max_iter=1000),
            df.iloc[tr],
            X_tr,
            y_tr,
            clusters,
            n_splits=5,
            cv = cv
        )
        if plot_importance:
            plt = plot_cluster_importance(imp_c)
        # Determine how many features to pick from each cluster 
        props = imp_c["mean"] / imp_c["mean"].sum()
        ns = np.round(props * K).astype(int)
        
        # Select important features and create reduced datasets
        sub_features = select_k_features(partitions, ns)
    else:
        # Select only the top feature w.r.t. importance from each cluster
        sub_features = (partitions.sort_values("importance", ascending=False)
                                  .groupby("cluster")
                                  .head(1)['feature'])
    return sub_features, ns, (plt, imp_c)


def scfm(part_tr, X_tr_norm, X_te_norm, y_tr,  agg, plot=False, verbose=False, seed=None):
    """Sub Cluster Feature Merging."""
    
    # Compute subclusters
    X_tr_sc = {}
    X_te_sc = {}
    n_cl = np.unique(part_tr["cluster"])
    for c in ( tqdm(n_cl) if verbose else n_cl ):
        ## Get features in cluster.
        f_in_c = part_tr.query(f"cluster == {c}")["feature"].values        
        ## Clean corr matrix
        C_tr_star_clean = clean_matrix_(X_tr_norm[f_in_c])
        ## Compute subclusters.
        sub_part_tr, _ = runN_Louvain(C_tr_star_clean, N=100, verbose=False, random_state=seed)
        ## Get original feature names back
        sub_part_tr["feature"] = part_tr.query(f"cluster == {c}")["feature"].values
        
        for sc in np.unique(sub_part_tr["cluster"]):
            f_sc = sub_part_tr.query(f"cluster == {sc}")["feature"].values
            X_tr_sc[f"cl{c}_subcl{sc}"] = agg(X_tr_norm[f_sc], axis=1)
            X_te_sc[f"cl{c}_subcl{sc}"] = agg(X_te_norm[f_sc], axis=1)
            
    # Convert dict to pandas df
    X_tr_sc = pd.DataFrame(X_tr_sc)
    X_te_sc = pd.DataFrame(X_te_sc)
    
    return X_tr_sc, X_te_sc


def preprocess(df, X_tr, X_te, y_tr, tr, comb, method=1):
    """
    Full preprocessing pipeline.
    method=1 is cluster weighted feature selection.
    method=2 is subcluster feature merging
    """
    # Impute missing values with the mean only (no bias).
    strategy = comb["impute_strategy"]
    X_tr, imputer = impute(X_tr, strategy)
    X_te, _ = impute(X_te, strategy, imputer=imputer)
    
    # Compute feature importances using Gradient Boosting Trees.
    lgbm_all_f = LGBMClassifier().fit(X_tr, y_tr)
    imp_tr = lgbm_all_f.feature_importances_
     
    # Clustering step.
    ## Compute and clean correlation matrix.
    ### Normalize before computing corr matrix
    X_tr_mean, X_tr_std = X_tr.mean(), X_tr.std()
    X_tr_norm = (X_tr - X_tr_mean) / X_tr_std
    X_te_norm = (X_te - X_tr_mean) / X_tr_std
    
    C_tr_clean = clean_matrix_(X_tr_norm)
    
    ## Compute clusters by running Louvain N times.
    n_runs = 100
    part_tr, G = runN_Louvain(C_tr_clean, N=n_runs, verbose=True)
    
    ## Attribute importance to each feature in clusters.
    part_tr["feature"] = ["feature_"+str(part_tr.index[i]) for i in range(len(part_tr))]
    part_tr["importance"] = imp_tr
    
    if method==1:  
        # Compute cluster importance, use to pick features
        K = comb["K"]
        sub_features, plt = cwfs(
            K = K,
            partitions = part_tr,
            df = df,
            X_tr = X_tr,
            y_tr = y_tr,
            tr = tr
        )

        X_tr_sc = X_tr[sub_features]
        X_te_sc = X_te[sub_features]
        
    elif method==2:
        # Compute new features from merging subclusters.
        agg = comb["agg"]
        X_tr_sc, X_te_sc = scfm(part_tr=part_tr,
                                X_tr_norm=X_tr_norm,
                                X_te_norm=X_te_norm,
                                y_tr = y_tr,
                                agg=agg)
    
    return X_tr_sc, X_te_sc, lgbm_all_f
