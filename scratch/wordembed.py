from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

model = word2vec.Word2Vec.load('data/model_300dim.pkl')
smiles = ['Br.Br.NCC1CCCN1c1cccnn1', "C#CCOc1cccc(CN)c1.Cl"]

for ismiles in smiles:
    
    mol = Chem.MolFromSmiles(ismiles)
    isentence = MolSentence(mol2alt_sentence(mol, radius = 2))
    iembedding = sentences2vec(isentence, model)
    iembedding = pd.DataFrame(iembedding)
    print(iembedding.shape)
    ipools = iembedding.mean(axis = 0).values
    ipools = np.append(ipools, iembedding.max(axis = 0).values)
    ipools = np.append(ipools, iembedding.sum(axis = 0).values)
    
    
    # DfVec(np.reshape(np.array(ivec[0]), (1,-1))).vec
    # np.array([x.vec for x in iembedding])
    
    # np.any(np.array([x in w2v_model.wv.index_to_key for x in isentence.sentence]))
    
    
    # model.wv.word_vec(model.wv.index_to_key[1873])
