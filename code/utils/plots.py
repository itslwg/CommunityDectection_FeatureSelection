import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from plotnine import *
from matplotlib.colors import ListedColormap


def plot_cluster_importance(imp, 
                            show: bool=True,
                            path="../figures/cluster_importance.pdf"):
    """Bar-plot MDA cluster importance."""
    imp = imp.sort_values(by="mean", ascending=True)
    cls = imp["index"].to_list()
    imp.loc[:, "index"] = pd.Categorical(imp["index"], categories=cls)
    plt = (
        ggplot(imp) + 
        geom_col(aes(x="index", y="mean")) + 
        scale_x_discrete(limits=cls) +
        coord_flip() + 
        ylab("Mean Decrease in Utility") + 
        xlab("Cluster") + 
        theme_matplotlib()
    )
    if show: 
        fig = plt.draw()
        fig.show()
    if path: plt.save(path, verbose=False)
    return plt
    

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object.
    
    Source:
        
    """
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(1, n_splits +1)) + ['Target', 'Date']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample row', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    # ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


def plot_clusters(partitions, G, save=True, path="../figures/louvain_graph.pdf"):
    """Plot clusters / communities from the Louvain algo."""
    fig = plt.figure(figsize=(12,8))
    pos = nx.spring_layout(G) # Slow
    # Color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partitions["cluster"].values + 1))
    nx.draw_networkx_nodes(G, pos, partitions.index, node_size=50,
                           cmap=cmap, node_color=list(partitions["cluster"].values))

    nx.draw_networkx_edges(G, pos, alpha=0.1)
    if save:
        plt.savefig(path)
    plt.show()       


def plot_merged_subclusters(imp, fnames, save=True, path="../figures/feature_importance_subclusters.pdf"):
    """Plot the merged subclusters"""
    imp_idx = np.argsort(imp)
    plt.bar(fnames[imp_idx], imp[imp_idx], color="black")
    plt.axhline(imp.mean(), linestyle="--")
    plt.xticks(rotation="vertical")
    plt.ylabel("Importance")
    plt.tight_layout()

    if save:
        plt.savefig(path)
    plt.show()
