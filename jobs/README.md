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

gcloud batch jobs delete fit --location us-central1
gcloud batch jobs submit fit --location us-central1 --config jobs/2-train-model/instance-template.json

gcloud batch jobs delete replaceblockids --location us-central1
gcloud batch jobs submit replaceblockids --location us-central1 --config jobs/replace-block-ids/instance-template.json

gcloud batch jobs delete submit --location us-central1
gcloud batch jobs submit submit --location us-central1 --config jobs/submit/instance-template.json

```

Helpful links and commands:

* https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs

gcloud compute accelerator-types list --filter="us-central1-a"
gcloud compute accelerator-types list --filter="us-west1-a"
gcloud compute regions list --filter="us"


