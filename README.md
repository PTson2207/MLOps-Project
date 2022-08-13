<h1>Machine Learning Operator

### Virtual environment
``` bash
python3 -m venv env-mlops-project
source env-mlops-project/Scripts/activate (window)
python -m pip install --upgrade pip setuptools wheel
```

### How to load file from another folder
```
improt sys
sys.path.insert(1, "./configs") #configs: folder configs in file
```

### How to run app by uvicorn
``` bash
uvicorn app.api:app \       # location of app (`app` directory > `api.py` script > `app` object)
    --host 0.0.0.0 \        # localhost
    --port 8000 \           # port 8000
    --reload \              # reload every time we update
    --reload-dir tagifai \  # only reload on updates to `tagifai` directory
    --reload-dir app        # and the `app` directory

```
```bash
# want to manage multiple uvicorn workers to enable parallelism in our application
gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app

```

### How to run DVC
```bash
dvc init
dvc remote storage add -d stores/blob # create folder blob in stores: where dvc storage
```
