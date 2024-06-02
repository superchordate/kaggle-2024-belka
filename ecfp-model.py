# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train, run_val
from modules.score import kaggle_score

# run model training on the train/train samples.
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, net, labels, scores = train(
        get_loader('out/train/train/', protein_name, n_files = 1),
        save_folder = 'out/train/train/',
        save_name = f'net-{protein_name}'
    )
    del net, labels, scores

ids, labels, scores = run_val(get_loader('out/train/val/', protein_name), net)

solution = pd.DataFrame({'protein_name': [protein_name]*len(labels)})
solution['split_group'] = 1
solution['id'] = 1
solution['binds'] = labels

submission = pd.DataFrame({'binds': scores})
submission['id'] = 1

kaggle_score(solution, submission, 'id')
