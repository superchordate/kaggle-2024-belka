from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import numpy as np

def get_pca(X, info_cutoff = 0.95, col_increment = None, verbose = 1, from_full = True):
    
    ncols = len(X[0])
    nrows = len(X)
    if col_increment == None: col_increment = int(.01*ncols) if ncols > 100 else 1
    if verbose > 0: print(f'running PCA on {ncols} columns {nrows} rows, col_increment: {col_increment}')
    if from_full and ncols < nrows:
        print('from_full is True but ncols < nrows, switching to from_full = False')
        from_full = False

     # scale and normalize prior to PCA.
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('normalize', Normalizer())
    ]).fit(X)

    X = pipe.transform(X)

    if from_full:

        # start at number cols minus one and drop columns until you get to info_cutoff% of the information.    
        n_components = ncols
        last_pca = None
        if n_components > 1 and n_components < nrows:

            n_components = n_components - col_increment

            while True:
                pca = PCA(n_components = n_components)
                pca = pca.fit(X)
                if np.sum(pca.explained_variance_ratio_) < info_cutoff:
                    break
                else:
                    if verbose > 1: print(f'cols: {n_components} explained: {round(np.sum(pca.explained_variance_ratio_), 2)}')
                    n_components = n_components - col_increment
                    last_pca = pca
                    
            # fit the final pca.
            n_components += col_increment
            pca = last_pca if (last_pca != None) else PCA(n_components = n_components).fit(X)

    else:

            # start at col_increment and increase until you get to info_cutoff% of the information.
            n_components = col_increment
            while True:
                pca = PCA(n_components = n_components)
                pca = pca.fit(X)
                if n_components >= ncols: n_components = ncols
                if (np.sum(pca.explained_variance_ratio_) > info_cutoff) or (n_components >= ncols):
                    break
                else:
                    if verbose > 1: print(f'cols: {n_components} explained: {round(np.sum(pca.explained_variance_ratio_), 2)}')
                    n_components = n_components + col_increment
        
    print({
        'starting cols': ncols, 
        'ending cols': n_components, 
        'explained': round(np.sum(pca.explained_variance_ratio_), 2)
    })

    pipe = Pipeline([
        ('scale', pipe['scale']),
        ('normalize', pipe['normalize']),
        ('pca', pca)
    ])
    
    return pipe

