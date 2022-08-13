#base image
FROM python:3.7-slim

#install dependencies
WORKDIR D:/MLOps-Project
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

#COPY
COPY  tagifai tagifai
COPY app app
COPY data data
COPY configs configs
COPY stores stores

#Pull assets from S3
RUN dvc init --no-scm
RUN dvc remote add -d storage stores/blob
RUN dvc pull

#export ports
EXPOSE 8000

#run app
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]
