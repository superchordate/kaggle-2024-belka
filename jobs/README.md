This folder contains code for running jobs on Google Cloud Platform (GCP). I (Bryce) have been using these to run operations too big for my laptop. See jobs/README.md for more information.

Each script is also set up so it can run locally, but typically on sampled data. 

Right now, only Bryce has access to deploy these to GCP so let him know if you'd like to attempt this.

To deploy a job: 

* Install the gcloud utility and set it up.
* Confirm you are on the correct project by running `gcloud config set project data-science-417721`
* Load the batch_job.py file you want to run to the kaggle storage bucket.

Use these scripts to delete/deploy jobs:
```
gcloud config set project data-science-417721

gcloud batch jobs delete sample --location us-central1
gcloud batch jobs submit sample --location us-central1 --config jobs/1-sample-data/instance-template.json

gcloud batch jobs delete preprocess --location us-central1
gcloud batch jobs submit preprocess --location us-central1 --config jobs/2-preprocess-data/instance-template.json

gcloud batch jobs delete fix-sample --location us-central1
gcloud batch jobs submit fix-sample --location us-central1 --config jobs/2b-fix-sample/instance-template.json

gcloud batch jobs delete fit --location us-central1
gcloud batch jobs submit fit --location us-central1 --config jobs/3-train-model/instance-template.json

gcloud batch jobs delete fit-voting --location us-central1
gcloud batch jobs submit fit-voting --location us-central1 --config jobs/3-train-voting/instance-template.json
```

Helpful links and commands:

* https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs

gcloud compute accelerator-types list --filter="us-central1-a"
gcloud compute accelerator-types list --filter="us-west1-a"
gcloud compute regions list --filter="us"


