import os, pickle, shutil

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

def pad0(x):    
    if x < 10: x = '0' + str(x)
    return x

def listfiles(dir, pattern):
    return [dir + '/' + f for f in os.listdir(dir) if pattern in f]