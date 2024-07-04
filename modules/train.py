import torch, os, time, gc, math
import torch.nn as nn
import numpy as np
import polars as pl
from modules.utils import device, gcp, fileexists, cloud
from modules.features import features
from modules.datasets import get_loader
from modules.nets import MLP_sm, MLP_md, MLP_lg, Siamese, ContrastiveLoss
   
def train(
        indir, 
        save_folder,
        save_name,        
        options,
        print_batches = 2000,
        model_load_path = None, 
        optimizer = None
):  
    idevice = device()
    if cloud(): print(idevice)

    # load mols and blocks.
    molpath = f'{indir}/mols.parquet' if not cloud() else 'mols.parquet'
    print(f'loading {molpath}')
    mols = pl.read_parquet(
        molpath, 
        columns = ['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index', 'binds_sEH', 'binds_BRD4', 'binds_HSA']
    )
    blockspath = f'out/blocks-3-pca.parquet' if not cloud() else 'blocks-3-pca.parquet'
    blocks = pl.read_parquet(blockspath, columns = ['index', 'features_pca'])

    # get the network, optimizer, and criterion.
    net, optimizer = get_model_optimizer(options, mols = mols, blocks = blocks, load_path = model_load_path)
    net = net.to(idevice)
    net = net.train()
    
    if not optimizer:
        #optimizer = optim.SGD(net.parameters(), lr=options['lr'], momentum=options['momentum'])
        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr = options['lr'], 
            weight_decay = 1e-5,
            amsgrad = True,
            fused = idevice == 'cuda'
        )

    if options['network'] == 'siamese': 
        criterion1 = ContrastiveLoss().to(idevice)
    else:
        criterion1 = nn.BCELoss().to(idevice)
        criterion2 = nn.BCELoss().to(idevice)
        criterion3 = nn.BCELoss().to(idevice)
        #criterion = nn.CrossEntropyLoss().to(idevice)

    # the data is too large to fit in memory, so we need to load it in batches.
    if options['n_rows'] != 'all':
        mols = mols.sample(options['n_rows'])
        print(f'sampled to {mols.shape[0]/1000/1000:.1f}M rows.')

    if 'rebalanceto' in options:

        # duplicate molecule ids that bind, to get the desired percentage.
        mols_binds_indexes = mols.filter(
            pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA')
        )
        current_pct = mols_binds_indexes.shape[0] / mols.shape[0]
        duplicate_count = math.ceil(options['rebalanceto'] / current_pct)
        print(f'rebalance: current {current_pct:.3f}, repeating {duplicate_count}x to reach {options["rebalanceto"]:.2f}')
        
        for i in range(duplicate_count):
            mols = pl.concat([mols_binds_indexes, mols])

        # shuffle the order so we don't have all the binding targets at the front. 
        index = np.random.permutation(np.array(range(mols.shape[0])))
        mols = mols.with_columns(pl.Series('index', index)).sort('index').drop('index')
        del mols_binds_indexes, current_pct, duplicate_count, i, index
        print(f'now {mols.shape[0]/1000/1000:.1f}M rows.')
    
    mols_binds_indexes = mols.filter(pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA'))
    print(f'{mols.shape[0]/1000/1000:.1f}M rows {mols_binds_indexes.shape[0]/mols.shape[0]:.3f} binds')
    mols = mols.with_columns(pl.Series('group', np.random.choice(range(options['num_splits']), mols.shape[0])))
    mols = mols.partition_by('group', include_key = False)
    print(f'split mols to {len(mols)} random splits for processing.')
    
    print(f'training {save_name}')
    devicesprinted = False
    notimprovedct = 0
    minloss = 9999999
    for epoch in range(options['epochs']):

        molct = 0
        for imols in mols:

            molct += 1
            print(f'epoch {epoch + 1} split {molct} of {len(mols)}')
            
            loader = get_loader(indir = '', mols = imols, blocks = blocks, options = options)
            print(f'{len(loader):,.0f} batches')
            
            start_time = time.time()
            loss = 0.0
            scores = {'sEH': [], 'BRD4': [], 'HSA': []}
            labels = {'sEH': [], 'BRD4': [], 'HSA': []}
            for i, data in enumerate(loader, 0):
            
                if options['network'] == 'siamese':
                    imolecule_ids, iX1, iX2, iX3, iy = data
                else:
                    imolecule_ids, iX1, iy = data
                
                # sometimes 1 row comes through, which causes an error.
                if iX1.shape[0] <= 1: continue
                
                optimizer.zero_grad()

                # if not devicesprinted:
                #     print(f'iX: {iX.type()} {iX.device}, iy: {iy["sEH"].type()} {iy["sEH"].device}, net: {next(net.parameters()).device}')
                #     devicesprinted = True
                
                if options['network'] == 'siamese':
                    outputs1, outputs2, outputs3 = net(iX1, iX2, iX3)
                    iloss = criterion1(outputs1, outputs2, outputs3, iy[options['protein']])
                else:
                    outputs1 = net(iX1)                
                    loss1 = criterion1(outputs1['sEH'], iy['sEH'])
                    loss2 = criterion2(outputs1['BRD4'], iy['BRD4'])
                    loss3 = criterion3(outputs1['HSA'], iy['HSA'])                    
                    iloss = loss1 + loss2 + loss3

                iloss.backward()
                optimizer.step()

                loss += float(iloss.cpu().item())

                doproteins = [options['protein']] if options['network'] == 'siamese' else iy.keys()
                for protein_name in doproteins:
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                    if options['network'] == 'siamese':
                        scores[protein_name] = np.append(scores[protein_name], outputs1.cpu().tolist())
                    else:
                        scores[protein_name] = np.append(scores[protein_name], outputs1[protein_name].cpu().tolist())
                del doproteins

                if (i % print_batches == 0) and (i != 0):
                    print(f'batch {i}, loss: {loss:.2f} {(time.time() - start_time)/60:.1f} mins')
                    if not cloud(): save_model(net, optimizer, save_folder, save_name, verbose = False)
                    torch.cuda.empty_cache()
                    start_time = time.time()
                    # if idevice == 'cuda': print(f'cuda memory allocated: {torch.cuda.memory_allocated(idevice)/1024/1024:.1f} GB')

                    if loss < minloss:
                        minloss = loss
                        notimprovedct = 0 
                    else:
                        notimprovedct += 1
                        print(f'notimproved: {notimprovedct}')
                        if notimprovedct >= options['early_stopping_rounds']:
                            print(f'loss has not improved in {options["early_stopping_rounds"]} rounds, stopping.')
                            save_model(net, optimizer, save_folder, save_name, verbose = True)
                            return net, labels, scores
                    loss = 0.0

                del i, data, imolecule_ids, iX1, iy, outputs1, iloss
                if options['network'] == 'siamese': 
                    del iX2, iX3, outputs2, outputs3
                else:
                    del loss1, loss2, loss3
            
            del imols, loader
            gc.collect()
            torch.cuda.empty_cache()
            if cloud(): save_model(net, optimizer, save_folder, save_name, verbose = False)
            net.to(idevice)
            net.train()
        
        if not cloud(): save_model(net, optimizer, save_folder, save_name, verbose = cloud())
        net.to(idevice)
        net.train()
        
    return net, labels, scores
    
def run_val(loader, net, options, print_batches = 2000): 

    print(f'{len(loader)} batches')

    net = net.eval()

    with torch.no_grad():
        scores = {'sEH': [], 'BRD4': [], 'HSA': []}
        labels = {'sEH': [], 'BRD4': [], 'HSA': []}
        molecule_ids = []
        for i, data in enumerate(loader, 0):
            
            if options['network'] == 'siamese':
                imolecule_ids, iX1, iX2, iX3, iy = data
                outputs = net(iX1, iX2, iX3)
            else:
                imolecule_ids, iX1, iy = data
                outputs = net(iX1)
            
            for protein_name in outputs.keys():
                if len(iy[protein_name]) > 0: # will be empty if this is a test set.
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                scores[protein_name] = np.append(scores[protein_name], outputs[protein_name].cpu().tolist())

            molecule_ids = np.append(molecule_ids, imolecule_ids)
            if (i % print_batches == 0) and (i != 0):
                print(f'batch {i}')
            del i, data, imolecule_ids, iX1, iy, outputs
            if options['network'] == 'siamese': del iX2, iX3
            
    return molecule_ids, labels, scores

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_model(model, optimizer, folder, name, verbose = True):

    # JIT is preferred but has had issues.
    model = model.cpu().eval()
    basepath = f'{folder}/{name}' if not cloud() else name

    # try:

    #     torch.save(model, f'{basepath}.torch')
    #     if gcp(): os.system(f'gsutil cp {basepath}.torch gs://kaggle-417721/{basepath}.torch')

    #     # model_scripted = torch.jit.script(model.cpu())
    #     # model_scripted.save(f'{basepath}.pt')        
    #     # if gcp(): os.system(f'gsutil cp {basepath}.pt gs://kaggle-417721/{basepath}.pt')
    #     if verbose:  print(f'saved {basepath}')

    # except:        
    #     print('!!failed to save model, load the weights instead.')

    # always save the model state just in case there is an error with another save method.
    torch.save(model.state_dict(), f'{basepath}.state')
    torch.save(optimizer.state_dict(), f'{basepath}-opt.state')
    if gcp(): 
        os.system(f'gsutil cp {basepath}.state gs://kaggle-417721/{basepath}.state')
        os.system(f'gsutil cp {basepath}-opt.state gs://kaggle-417721/{basepath}-opt.state')

    model = model.to(device())
    model = model.train()

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def get_model_optimizer(options, mols = None, blocks = None, load_path = None, load_optimizer = True):

    # if '.pt' in filename:

    #     model = torch.jit.load(filename)

    # elif '.torch' in filename:

    #     model = torch.load(filename)

    # elif '.tweights' in filename:
    if options['network'] == 'siamese':
        input_len = len(features(mols[0,], blocks, options)[0][0])
    else:
        input_len = len(features(mols[0,], blocks, options)[0])
    
    if options['network'] == 'lg':
        model = MLP_lg(options = options, input_len = input_len)
    elif options['network'] == 'md':
        model = MLP_md(options = options, input_len = input_len)
    elif options['network'] == 'sm':
        model = MLP_sm(options = options, input_len = input_len)
    elif options['network'] == 'siamese':
        model = Siamese(options = options, input_len = input_len)
    else:
        raise ValueError('network must be lg, md, or sm ({options["network"]})')

    if load_path is not None: 
        print(f'loading model path: {load_path}.state')
        model.load_state_dict(torch.load(f'{load_path}.state'))
    else:
        print('starting new model')

    model = model.to(device())
        
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr = options['lr'], 
    #     weight_decay = 1e-5,
    #     amsgrad = True,
    #     fused = device() == 'cuda'
    # )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=options['lr'], momentum=options['momentum'])

    if load_path is not None and load_optimizer and fileexists(f'{load_path}-opt.state'): 
        print(f'loading optimizer {load_path}-opt.state')
        optimizer.load_state_dict(torch.load(f'{load_path}-opt.state'))

    return model, optimizer

