import pickle, sys, os
import polars as pl
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import xgboost as xgb
from sklearn.metrics import average_precision_score

protein = 'sEH' # 'sEH', 'BRD4', 'HSA'
just_testing = False

def dircreate(x): 
    if not os.path.exists(x):
        os.makedirs(x)

def load1(path):
    with open(path, 'rb') as pkl_file:
        obj = pickl

if len(sys.argv) > 1:
    protein = sys.argv[1]

indir = 'out/test/test/mols/'
outdir = 'out/test/test/similarity-scores/'
m_xgb = load1(f'out/train/xgb-{protein}.pkl')
batch_files = [indir + x for x in os.listdir(indir) if protein in x]
dircreate(outdir)
print(f'{protein} {len(batch_files)} batches')

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

ct = 0
fpgen = AllChem.GetRDKitFPGenerator().GetFingerprint
for ifile in batch_files:
    mol_supp = Chem.SDMolSupplier(ifile)
    scores = []
    ct+= 1
    print(f'{protein} batch {ct} records {len(mol_supp)}')
    ct_mol = 0
    for mol in mol_supp:
        ct_mol += 1
        if (ct_mol % 500 == 0) and (ct_mol > 0): print(f'{ct_mol}')
        if ct_mol > 5 and just_testing: break
        scores.append(get_scores(mol, protein = protein))
    scores = np.array(scores, dtype=np.float32)
    proba = np.array([x[1] for x in m_xgb.predict_proba(scores)])
    results = pd.DataFrame(scores)
    results['proba'] = proba
    pl.DataFrame(results).write_parquetc
    if just_testing: break
