from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def ecfp(smile, radius=2, bits=1024):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        return np.full(bits, -1)
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))
