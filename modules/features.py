import numpy as np
import polars as pl
import pandas as pd
from modules.utils import cloud

if not cloud():

    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from pysmilesutils.tokenize import SMILESTokenizer
    from rdkit.Chem import rdFingerprintGenerator
    from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
    from gensim.models import word2vec
    import networkx as nx
    from karateclub import Graph2Vec

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius =2 , fpSize = 2048)
    w2v_model = word2vec.Word2Vec.load('data/model_300dim.pkl')
    graph_embed_model = Graph2Vec()

def ecfp(smile):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        return np.full(2048 * 2, -1)
    return list(mfpgen.GetFingerprint(molecule)) + list(mfpgen.GetCountFingerprint(molecule))

def blocks_add_ecfp(blocks):
    iblocks = blocks.select(['index', 'smiles'])
    iblocks = iblocks.map_rows(lambda row: (row[0], ecfp(row[1])))
    iblocks.columns = ['index', 'ecfp'] # prior operation will remove column names.
    return blocks.with_columns(iblocks['ecfp'])

def pools(x):
    xDF = pd.DataFrame(np.array(x))
    ipools = xDF.mean(axis = 0).values
    ipools = np.append(ipools, xDF.max(axis = 0).values)
    ipools = np.append(ipools, xDF.sum(axis = 0).values)
    return ipools

def blocks_add_onehot(blocks):

    # fit the tokenizer.
    tokenizer = SMILESTokenizer(encoding_type = "one hot")
    tokenizer.create_vocabulary_from_smiles(blocks['smiles'])

    # create the onehot encoding.
    smiles_encoded = tokenizer(blocks['smiles'], enclose = False)
    smiles_encoded_pools = [pools(x) for x in smiles_encoded]
    smiles_encoded_pools = np.vstack(smiles_encoded_pools)
    
    return blocks.with_columns(pl.Series('onehot', smiles_encoded_pools))

# https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html
# len(getMolDescriptors(Chem.MolFromSmiles(all_blocks['smiles'][0]))) # 210
#warnings.filterwarnings("ignore", category=DeprecationWarning)
def getMolDescriptors(smiles, missingVal=-1):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return list(np.full(210, -1)) # 12 metrics.
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(molecule)
        except:
            val = missingVal
        res[nm] = val

    return [float(res[x]) for x in res]

# def descriptors(smile):
#     molecule = Chem.MolFromSmiles(smile)    
#     if molecule is None:
#         return []
#         return list(np.full(12, -1)) # 12 metrics.
#     return [
#         Descriptors.ExactMolWt(molecule),
#         Descriptors.MolWt(molecule),
#         # Descriptors.FpDensityMorgan1(molecule),
#         # Descriptors.FpDensityMorgan2(molecule),
#         # Descriptors.FpDensityMorgan3(molecule),
#         Descriptors.HeavyAtomMolWt(molecule),
#         Descriptors.MaxAbsPartialCharge(molecule),
#         Descriptors.MaxPartialCharge(molecule),
#         Descriptors.MinAbsPartialCharge(molecule),
#         Descriptors.MinPartialCharge(molecule),
#         Descriptors.NumRadicalElectrons(molecule),
#         Descriptors.NumValenceElectrons(molecule),
#     ]

def blocks_add_descriptors(blocks):
    return blocks.with_columns(
        pl.Series('descrs', [getMolDescriptors(smiles) for smiles in blocks['smiles']])
    )

def features(dt, blocks, options):

    blocks = blocks.with_columns(pl.col('index').cast(pl.UInt16))
    idt = dt.join(blocks, left_on = 'buildingblock1_index', right_on = 'index', how = 'inner', suffix = '1') \
        .join(blocks, left_on = 'buildingblock2_index', right_on = 'index', how = 'inner', suffix = '2') \
        .join(blocks, left_on = 'buildingblock3_index', right_on = 'index', how = 'inner', suffix = '3')

    features = idt['features_pca'].list.concat(
        idt['features_pca2'].list.concat(
            idt['features_pca3']
    ))

    return(np.vstack(features))

# https://towardsdatascience.com/basic-molecular-representation-for-machine-learning-b6be52e9ff76
# download from https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl.
def word_embedding(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(1024, -1)
    
    isentence = MolSentence(mol2alt_sentence(mol, radius=1))
    iembedding = sentences2vec(isentence, w2v_model) # this will be 300 cols and n rows (n = number of words).

    return pools(iembedding)

def blocks_add_word_embeddings(blocks):
    word_embeddings = [word_embedding(smiles) for smiles in blocks['smiles']]
    return blocks.with_columns(pl.Series('word_embeddings', word_embeddings))

# graph embeddings:
# https://towardsdatascience.com/basic-molecular-representation-for-machine-learning-b6be52e9ff76
# define the function for coverting rdkit object to networkx object
def mol_to_nx(smiles):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    if mol is None:
        return G
    
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
        
    return G

def blocks_add_graph_embeddings(blocks):
    graphs = [mol_to_nx(x) for x in blocks['smiles']]
    graph_embed_model.fit(graphs)
    hiv_graph2vec = graph_embed_model.get_embedding()
    return blocks.with_columns(pl.Series('graph_embeddings', hiv_graph2vec))

