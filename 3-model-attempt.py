import pickle, sys
import polars as pl
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import xgboost as xgb
from sklearn.metrics import average_precision_score

protein = 'sEH' # 'sEH', 'BRD4', 'HSA'
just_testing = False

if len(sys.argv) > 1:
    protein = sys.argv[1]

# pick a random molecule from the validation set.
print(f'get scores for {protein}')
val_supp = Chem.SDMolSupplier(f'out/train/val/mols/{protein}-sample.sdf')
fpgen = AllChem.GetRDKitFPGenerator().GetFingerprint

# print('find binds')
# binds = np.where(val['binds'] == 1)[0]
# not_binds = np.where(val['binds'] == 0)[0]

# calculate metrics comparing the molecule to the training set.
def get_scores(mol, protein):
    fps = fpgen(mol)
    ms = []
    with Chem.MultithreadedSDMolSupplier(f'out/train/train/mols/{protein}-sample.sdf') as sdSupl:
        for imol in sdSupl:
            if imol is not None:
                ifps = fpgen(imol)
                ms.append({
                    'dice': DataStructs.DiceSimilarity(fps, ifps),
                    'tanimoto': DataStructs.TanimotoSimilarity(fps, ifps),
                    'tversky': DataStructs.TverskySimilarity(fps, ifps, 0.5,0.5),
                    'cosine': DataStructs.CosineSimilarity(fps, ifps),
                    'sokal': DataStructs.SokalSimilarity(fps, ifps),
                    'russel': DataStructs.RusselSimilarity(fps, ifps),
                    'kulczynski': DataStructs.KulczynskiSimilarity(fps, ifps),
                    'mcconnaughey': DataStructs.McConnaugheySimilarity(fps, ifps)
                })        
    return pl.DataFrame(ms).mean().to_numpy()[0]

scores = []
print(f'{len(val_supp)} records')
ct = 0
for mol in val_supp:
    if (ct % 100 == 0) and (ct > 0): print(f'{ct}')
    scores.append(get_scores(mol, protein = protein))    
    ct+= 1
    if just_testing and (ct > 5): break
    
# scores_binds = get_scores(val_supp[int(binds[0])], protein = 'sEH')

# scores_not = get_scores(val_supp[int(not_binds[0])], protein = 'sEH')

# print(np.concatenate([
#     np.reshape(, (-1,1)),
#     np.reshape(scores_not.mean().to_numpy(), (-1,1)),
# ], axis = 1))

# build a modelwith the data. 
print('training model')
y = pl.read_parquet(f'out/train/val/base-sample-{protein}.parquet')['binds'].to_numpy()
X = np.array(scores)

if just_testing: y = y[range(len(X))]

m_xgb = xgb.XGBClassifier(max_depth = 3, subsample = 0.5, gamma = 2).fit(X, y)

print(average_precision_score(y, m_xgb.predict(X)))

def save1(obj, path):
    output = open(path, 'wb')
    pickle.dump(obj, output)
    output.close()

save1(m_xgb, f'out/train/xgb-{protein}.pkl')

