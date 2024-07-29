# Welcome!

I've shared this repo to show some of my work in the data science realm. This code was only used on a solo project so it isn't super clean or well documented. That said, it'll give you an idea of what my code is like and what work I've done.

My preference in competitions is to build a neural network. These models easily train on very large and complex data and do a lot of the feature engineering and selection work for you. They don't always make sense in non-competition projects where explainability is important in addition to performance. 

Some interesting things to look at: 
* [jobs/2-train-model/](https://github.com/superchordate/kaggle-2024-belka/tree/main/jobs/2-train-model): I often use Batch Jobs on Google Cloud Projects to run projects that need GPU and high memory.
* [modules/](https://github.com/superchordate/kaggle-2024-belka/tree/main/modules) and [fun/](https://github.com/superchordate/kaggle-2024-belka/tree/main/fun): neat functions that I've written.
* [scripts/](https://github.com/superchordate/kaggle-2024-belka/tree/main/scripts): end-to-end, stepwise scripts from batched data intake (the data was very large), preprocessing, feature engineering, model training, and inference. 

## About Me

I'm an independent contractor helping companies build custom cloud apps and leverage data science, visual analytics, and AI. I offer low introductory rates, free consultation and estimates, and no minimums, so contact me today and let's chat about how I can help!

https://www.bryce-chamberlain.com/

## Installation

* Instally Python `3.8.4`
  - Choose option "Add python.exe to PATH"
  - Restart your computer after the install is finished. 
  - Take note of the location of your Python installation. For me it was `C:\Users\Bryce\AppData\Local\Programs\Python\Python311\`. Replace my path with yours below.
* Install [virtualenv via pip](https://virtualenv.pypa.io/en/latest/installation.html) with command `python -m pip install --user virtualenv`.
  - The script will suggest adding virtualenv.exe to PATH, do so now. For me it is: `C:\Users\Bryce\AppData\Roaming\Python\Python311\Scripts`
* Set up your local project virtual environment by running:
    - `virtualenv .venv -p "C:\Users\super\AppData\Local\Programs\Python\Python311\python.exe"`.
    - `".venv/Scripts/activate.bat"`
    - `pip install -r requirements.txt`

_You don't need to do this, but for my records, here is how the initial install was done:_
```python
pip install 'polars[numpy,pandas,pyarrow,timezone]'
pip freeze > requirements.txt
python -m venv .venv  -p "C:\Users\Bryce\AppData\Local\Programs\Python\Python311\python.exe"
```

* Install libaries from github.
```
pip install C:/Users/super/Documents/kaggle/mol2vec
pip install git+https://github.com/MolecularAI/pysmilesutils.git
```

## Folders

| Folder | Info |
| ------------- | ------------- |
| .venv | Python virtual environment files (you'll see this folder after you follow Installation above). |
| data | Data downloaded from sources (for now, just Kaggle competition data). |
| fun | Useful Python functions. Code at scripts\1-get-data.py line ~6 loads them, this code is used in a few different places. In some cases, we move the function code into scipts. |
| jobs | Code for running jobs on Google Cloud Platform. I (Bryce) have been using these to run operations too big for my laptop. See jobs/README.md for more information. |
| out | Data created as output from the code in this repo. |
| scratch | Temporary place for files that are unfinished or temporary. |
| scripts | Python scripts that form the main project. |

## Links

* [Kaggle Submission Notebook](https://www.kaggle.com/code/brycechamberlain/bryce-chamberlain-home-credit-submission)
* [GitHub](https://github.com/superchordate/kaggle-2024-belka)

### Instructions

* Run 1-train-test-split.py to create the initial train/test splits
