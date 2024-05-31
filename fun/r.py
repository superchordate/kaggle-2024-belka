# author: Bryce C
# R functions for python.

import numpy as np
import os, re
import pickle

def nrow(x): return x.shape[0]
def ncol(x): return x.shape[1]
def which(x): 
    return list(np.where(x)[0])
def stop(x): raise Exception(x)
def grepl(pattern, x, nafalse = True): 

    ire = re.compile(pattern)
    val = pd.Series([ire.search(ix) for ix in x])
    val[~val.isnull()] = True
    if nafalse: val[val.isnull()] = False
    return val.tolist()

def grep(pattern, x, naskip = True, value = False): 
    idx = which(grepl(pattern = pattern, x = x, nafalse = naskip))
    if value: 
        return [x[i] for i in idx]
    else:
        return idx
    
def trimws(x): return strip(x)
def setdiff( x, y ): return [ i for i in x if i not in y ]
def table(x): return x.value_counts( ascending = False )
def fileexists(x): return os.path.exists(x)
def dircreate(x): 
    if not os.path.exists(x):
        os.makedirs(x)

def savePkl(x, path):
    output = open(path, 'wb')
    pickle.dump(x, path)
    output.close()

def loadPkl(path):
    pkl_file = open(path, 'rb')
    x = pickle.load(pkl_file)
    pkl_file.close()
    return x

# https://stackoverflow.com/a/5409395/4089266
def unlist(x):
    flatten = lambda *n: (e for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    return list(flatten(x))

