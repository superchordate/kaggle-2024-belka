import os, pickle, shutil, torch, platform
import pyarrow.parquet as pq
import numpy as np

def dircreate(x, fromscratch = False): 
    if os.path.exists(x) and fromscratch: shutil.rmtree(x)
    if not os.path.exists(x): os.makedirs(x)

def unlist(x):
    flatten = lambda *n: (e for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    return list(flatten(x))

def save1(obj, path):
    output = open(path, 'wb')
    pickle.dump(obj, output)
    output.close()
    
def load1(path):
    with open(path, 'rb') as pkl_file:
        obj = pickle.load(pkl_file)
    return obj

def pad0(x, n = 3):
    if x < 10 and n >= 3: 
        x = '00' + str(x)
    elif x < 100 or (x < 10 and n == 2): 
        x = '0' + str(x)
    return x

def listfiles(dir, pattern = None):
    if pattern:
        return [dir + '/' + f for f in os.listdir(dir) if pattern in f]
    else:
        return [dir + '/' + f for f in os.listdir(dir)]

def write_parquet_from_pyarrow(x, path):
    writer = pq.ParquetWriter(path, x.schema)
    writer.write_table(x)
    writer.close()

def unlist_numpy(x):
    return np.reshape(x, (1,-1))[0]

def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def gcp():    
    return platform.system() != 'Windows'

def fileexists(x): 
    return os.path.exists(x)

def fileremove(x):
    if fileexists(x): os.remove(x)

