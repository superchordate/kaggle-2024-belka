# we want a random sample of molecules that bind for each protein
#  such that we can loop over it quickly in random order with Chem.MultithreadedSDMolSupplier('*.sdf').
#  for now, we'll work with just one protein, sEH.

import os, sys
import polars as pl
from rdkit import Chem
from modules.utils import dircreate, pad0

if len(sys.argv) == 2:
    
    protein = sys.argv[1] #sEH BRD4 HSA
    
    def get_mols(train_test, train_val, protein, sample = None, onlybinds = False, balance = None):
        
        indir = f'out/{train_test}/{train_val}/base/'
        dircreate(f'out/{train_test}/{train_val}/mols/')
        batch_files = [indir + x for x in os.listdir(indir)]
        print(f'{len(batch_files)} batches for {protein}')
        
        if sample: sample = int(sample/len(batch_files)) # sample per file.
        
        filename = f'{protein}{"-sample" if sample else ""}{"-binds" if onlybinds else ""}'
        
        ct = 0
        sample_dt = []
        for ifile in batch_files:
            with Chem.SDWriter(f'out/{train_test}/{train_val}/mols/{filename}-{pad0(ct)}.sdf') as w:
                
                ct+= 1
                idt = pl.read_parquet(ifile)
                
                if onlybinds:
                    idt = idt.filter(pl.col('protein_name') == protein, pl.col('binds') == 1)
                else: 
                    idt = idt.filter(pl.col('protein_name') == protein)
                
                if idt.shape[0] == 0:
                    continue
                
                if balance:
                    # for val, let's balance the binds a bit.
                    ibinds = idt.filter(pl.col('binds') == 1)
                    inot = idt.filter(pl.col('binds') == 0)
                    idt = pl.concat([ibinds, inot.sample(int(ibinds.shape[0]/balance))])
                
                if sample: 
                    if sample < idt.shape[0]: 
                        idt = idt.sample(sample)
                    sample_dt.append(idt)
                
                idt = idt.select(['id', 'molecule_smiles'])
                idt = idt.map_rows(lambda row: (row[0], Chem.MolFromSmiles(row[1])))
                for mol in idt['column_1']:
                    w.write(mol)
                
                print(f'{train_test} {train_val} batch {ct}')
                
        if sample: pl.concat(sample_dt).write_parquet(f'out/{train_test}/{train_val}/{filename}.parquet')
    
    get_mols('train', 'train', protein)
    get_mols('train', 'val', protein)
    
    # get_mols('train', 'train', protein, sample = 500)
    # get_mols('train', 'val', protein, sample = 1000)
    # get_mols('test', 'test', protein)
    
    def get_mols_by_batch_for_test(train_test, train_val, protein):
        indir = f'out/{train_test}/{train_val}/base/'
        batch_files = [indir + x for x in os.listdir(indir)]
        print(f'{len(batch_files)} batches for {protein}')
        ct = 0
        dircreate(f'out/{train_test}/{train_val}/mols/')
        for ifile in batch_files:
            ct+= 1
            idt = pl.read_parquet(ifile)
            idt = idt.filter(pl.col('protein_name') == protein)
            if idt.shape[0] == 0:
                continue
            idt = idt.select(['id', 'molecule_smiles'])
            idt = idt.map_rows(lambda row: (row[0], Chem.MolFromSmiles(row[1])))            
            with Chem.SDWriter(f'out/{train_test}/{train_val}/mols/{protein}-{"{0:0=2d}".format(ct)}.sdf') as w:
                for mol in idt['column_1']:
                    w.write(mol)
            print(f'{train_test} {train_val} batch {ct}')
    
    # for protein in ['sEH', 'BRD4', 'HSA']: # ['sEH', 'BRD4', 'HSA']
    #     get_mols_by_batch_for_test('test', 'test', protein)
        
