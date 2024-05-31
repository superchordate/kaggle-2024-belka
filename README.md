## Installation

* Instally Python [3.11.8](https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe). This is the [latest version of Python that works with Pytorch](https://pytorch.org/get-started/locally/)
  - Choose option "Add python.exe to PATH"
  - Restart your computer after the install is finished. 
  - Take note of the location of your Python installation. For me it was `C:\Users\Bryce\AppData\Local\Programs\Python\Python311\`. Replace my path with yours below.
* Install [virtualenv via pip](https://virtualenv.pypa.io/en/latest/installation.html) with command `python -m pip install --user virtualenv`.
  - The script will suggest adding virtualenv.exe to PATH, do so now. For me it is: `C:\Users\Bryce\AppData\Roaming\Python\Python311\Scripts`
* Set up your local project virtual environment by running:
    - `virtualenv .venv -p "C:\Users\Bryce\AppData\Local\Programs\Python\Python311\python.exe"`.
    - `".venv/Scripts/activate.bat"`
    - `pip install -r requirements.txt`

_You don't need to do this, but for my records, here is how the initial install was done:_
```python
pip install 'polars[numpy,pandas,pyarrow,timezone]'
pip freeze > requirements.txt
python -m venv .venv  -p "C:\Users\Bryce\AppData\Local\Programs\Python\Python311\python.exe"
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
| scripts | Python scripts that form the main project (right now just data prep/explore). |

## Links

* [Kaggle Submission Notebook](https://www.kaggle.com/code/brycechamberlain/bryce-chamberlain-home-credit-submission)
* [GitHub](https://github.com/superchordate/kaggle-2024-credit)
* [Data folder on Drive](https://drive.google.com/drive/folders/1aYeJOcdap4l3VKbYSHgFpxzfA3AoDRO3?usp=drive_link)

### Instructions for Sarah:

* Start with scripts\feature-engineering.py. 
* I've sampled the data down so it is easier to work with. Feel free to sample it down even more with the code at jobs\1-sample-data\batch-job.py
* See the comment at the top of scripts\feature-engineering.py for where to get the data you'll need. 
