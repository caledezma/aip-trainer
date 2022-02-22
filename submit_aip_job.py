"""Python script to submit a job to AIP using the python AIP"""
from googleapiclient import discovery, errors
import logging

training_inputs = {
    'masterConfig': {
        'imageUri': 'gcr.io/nplan-scheduling/carlos-aip-test-gpu:wandb'
    },
    'scaleTier': 'BASIC_GPU',
    'args': ['--epochs', '5', '--hidden-layers', '15', '--use-wandb-secret'],
    'region': 'us-central1',
    'serviceAccount': 'pipeline@nplan-scheduling.iam.gserviceaccount.com',
}

job_spec = {
    'jobId': 'train_from_py_1',
    'trainingInput': training_inputs,
}

project_name = 'nplan-scheduling'
project_id = 'projects/{}'.format(project_name)
cloudml = discovery.build('ml', 'v1')

request = cloudml.projects().jobs().create(body=job_spec,parent=project_id)

try:
    response = request.execute()

except errors.HttpError as err:
    logging.error('There was an error creating the training job. Check the details:')
    logging.error(err._get_reason())
