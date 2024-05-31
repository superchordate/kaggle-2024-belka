# container setup.
FROM google/cloud-sdk
COPY requirements.txt ./
RUN apt-get update && \
    apt-get install unzip zip && \
    pip install -r requirements.txt && \
    python3 --version && \
    pip freeze && \
    mkdir ./out

# copy the data, run the script, and save the result.
CMD df -h && \
    gsutil cp gs://kaggle-417721/batch-data.zip ./ && \
    gsutil cp gs://kaggle-417721/batch-job.py ./ && \ 
    unzip batch-data.zip && \
    python3 batch-job.py && \
    zip batch-output.zip ./out/ && \
    gsutil cp ./batch-output.zip gs://kaggle-417721/batch-output.zip
