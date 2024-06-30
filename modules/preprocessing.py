from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import numpy as np

def get_pca(X, info_cutoff = 0.95, col_increment = None, verbose = 1):
    
    ncols = len(X[0])
    nrows = len(X)
    if col_increment == None: col_increment = int(.01*ncols) if ncols > 100 else 1
    if verbose > 0: print(f'running PCA on {ncols} columns {nrows} rows, col_increment: {col_increment}')

     # scale and normalize prior to PCA.
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('normalize', Normalizer())
    ]).fit(X)

    X = pipe.transform(X)

    # run the middle point to see if we should be running from full or not.
    n_components = int(ncols/2)
    if n_components < nrows:
        raise Exception(f'n_components [{n_components}] must be less than the number of rows [{nrows}]')    
    pca, explained_variance_ratio = fit_return_result(X, n_components)
    from_full = explained_variance_ratio > info_cutoff
    if verbose > 1: print(f'ncols/2 explained {round(explained_variance_ratio, 2)}, using from_full = {from_full}')

    if from_full:

        last_pca = None
        if n_components > 1 and n_components < nrows:

            n_components = n_components - col_increment

            while True:
                pca, explained_variance_ratio = fit_return_result(X, n_components)
                if explained_variance_ratio < info_cutoff:
                    break
                else:
                    if verbose > 1: print(f'cols: {n_components} explained: {round(explained_variance_ratio, 2)}')
                    n_components = n_components - col_increment
                    last_pca = pca
                    
            # fit the final pca.
            n_components += col_increment
            pca = last_pca if (last_pca != None) else PCA(n_components = n_components).fit(X)

    else:

        while True:
            if n_components >= ncols: n_components = ncols
            pca, explained_variance_ratio = fit_return_result(X, n_components)
            if (explained_variance_ratio > info_cutoff) or (n_components >= ncols):
                break
            else:
                if verbose > 1: print(f'cols: {n_components} explained: {round(explained_variance_ratio, 2)}')
                n_components = n_components + col_increment
        
    print({
        'starting cols': ncols, 
        'ending cols': n_components, 
        'explained': round(explained_variance_ratio, 2)
    })

    pipe = Pipeline([
        ('scale', pipe['scale']),
        ('normalize', pipe['normalize']),
        ('pca', pca)
    ])
    
    return pipe

def fit_return_result(X, n_components):
    pca = PCA(n_components = n_components)
    pca = pca.fit(X)
    return pca, np.sum(pca.explained_variance_ratio_)
