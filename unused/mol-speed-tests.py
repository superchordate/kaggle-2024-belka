from modules.mols import get_blocks
# from chemml.chem import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem

 
building_blocks = get_blocks('train', just_testing = True)

fpgen = AllChem.GetMorganGenerator(radius=2)

# mol = Molecule(, input_type='smiles')
m1 = Chem.MolFromSmiles(building_blocks['smile'][0])


# fpgen.GetSparseCountFingerprint(m1)

list(AllChem.GetMorganFingerprintAsBitVect(m1, radius=3, nBits=10))

list(AllChem.EmbedMolecule(m1))

building_blocks['ecfp_pca'][0]


len(building_blocks['ecfp'][0])
