import torch, os, time, gc
import torch.nn as nn
import numpy as np
import polars as pl
from modules.utils import device, gcp
from modules.features import features
from modules.datasets import get_loader
from modules.nets import MLP_sm, MLP_md, MLP_lg
   
def train(
        indir, 
        save_folder,
        save_name,        
        options,
        print_batches = 2000,
        net = None, 
        optimizer = None,
        # criterion = nn.MSELoss()
        criterion = None
):  
    idevice = device()
    if gcp(): print(idevice)

    # load mols and blocks.
    molpath = f'{indir}/mols.parquet' if not gcp() else 'mols.parquet'
    print(f'loading {molpath}')
    mols = pl.read_parquet(
        molpath, 
        columns = ['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index', 'binds_sEH', 'binds_BRD4', 'binds_HSA']
    )
    blockspath = f'{indir}/blocks/blocks-3-pca.parquet' if not gcp() else 'blocks-3-pca.parquet'
    blocks = pl.read_parquet(blockspath, columns = ['index', 'features_pca'])

    # get the network, optimizer, and criterion.
    if not net:
        print('starting from clean network')
        input_len = len(features(mols[0,], blocks, options)[0])
        if options['network'] == 'lg':
            net = MLP_lg(options = options, input_len = input_len)
        elif options['network'] == 'md':
            net = MLP_md(options = options, input_len = input_len)
        else:
            net = MLP_sm(options = options, input_len = input_len)
        del input_len
    else:
        print('using existing network')
    net = net.to(idevice)
    
    if not optimizer:
        #optimizer = optim.SGD(net.parameters(), lr=options['lr'], momentum=options['momentum'])
        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr = options['lr'], 
            weight_decay = 1e-5,
            amsgrad = True,
            fused = gcp()
        )

    if not criterion: 
        criterion1 = nn.BCELoss().to(idevice)
        criterion2 = nn.BCELoss().to(idevice)
        criterion3 = nn.BCELoss().to(idevice)
        #criterion = nn.CrossEntropyLoss().to(idevice)

    # the data is too large to fit in memory, so we need to load it in batches.
    if options['n_rows'] == 'all':
        mols = mols.with_columns(pl.Series('group', np.random.choice(range(options['num_splits']), mols.shape[0])))
        mols = mols.partition_by('group', include_key = False)
        print(f'split mols to {options["num_splits"]} random splits for processing.')
    else:
        mols = [mols.sample(options['n_rows'])]
        print(f'sampled to {options["n_rows"]/1000/1000:.1f}M rows.')
    
    print(f'training {save_name}')
    for epoch in range(options['epochs']):

        molct = 0
        for imols in mols:

            molct += 1
            # print(f'epoch {epoch + 1} split {molct} of {options["num_splits"]}')
            
            loader = get_loader(indir = '', mols = imols, blocks = blocks, options = options)
            print(f'{len(loader):,.0f} batches')
            
            start_time = time.time()
            loss = 0.0
            scores = {'sEH': [], 'BRD4': [], 'HSA': []}
            labels = {'sEH': [], 'BRD4': [], 'HSA': []}
            for i, data in enumerate(loader, 0):
                
                imolecule_ids, iX, iy = data
                optimizer.zero_grad()

                print(f'iX: {iX.device}, iy: {iy["sEH"].device}, net: {next(net.parameters()).device}')

                outputs = net(iX)
                
                loss1 = criterion1(outputs['sEH'], iy['sEH'])
                loss2 = criterion2(outputs['BRD4'], iy['BRD4'])
                loss3 = criterion3(outputs['HSA'], iy['HSA'])
                
                iloss = loss1 + loss2 + loss3
                iloss.backward()
                optimizer.step()

                loss += iloss.cpu().item()
                for protein_name in iy.keys():
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                    scores[protein_name] = np.append(scores[protein_name], outputs[protein_name].cpu().tolist())

                if (i % print_batches == 0) and (i != 0):
                    print(f'batch {i}, loss: {loss:.0f} {(time.time() - start_time)/60:.1f} mins')
                    start_time = time.time()
                    loss = 0.0
                    save_model(net, save_folder, save_name, verbose = False)

                del i, data, imolecule_ids, iX, iy, outputs, loss1, loss2, loss3, iloss
            
            del imols, loader
            gc.collect()
        
        save_model(net, save_folder, save_name, verbose = gcp())
        return net, labels, scores
    
def run_val(loader, net, print_batches = 2000): 

    print(f'{len(loader)} batches')

    net = net.eval()

    with torch.no_grad():
        scores = {'sEH': [], 'BRD4': [], 'HSA': []}
        labels = {'sEH': [], 'BRD4': [], 'HSA': []}
        molecule_ids = []
        for i, data in enumerate(loader, 0):
            
            imolecule_ids, iX, iy = data
            outputs = net(iX)
            
            for protein_name in outputs.keys():
                if len(iy[protein_name]) > 0: # will be empty if this is a test set.
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                scores[protein_name] = np.append(scores[protein_name], outputs[protein_name].cpu().tolist())

            molecule_ids = np.append(molecule_ids, imolecule_ids)
            if (i % print_batches == 0) and (i != 0):
                print(f'batch {i}')
            del i, data, imolecule_ids, iX, iy, outputs
            
    return molecule_ids, labels, scores

def save_model(model, folder, name, verbose = True):
    model_scripted = torch.jit.script(model.cpu())
    if gcp(): 
        model_scripted.save(f'{name}.pt')
        os.system(f'gsutil cp {name}.pt gs://kaggle-417721/{name}.pt')
    else:
        model_scripted.save(f'{folder}/{name}.pt')
    if verbose: 
        print(f'saved {folder}/{name}.pt')
    model = model.to(device())
    model = model.train()

        