"""Python script to submit a job to AIP using the python AIP"""
import argparse
import os
import yaml

from dotenv import load_dotenv, find_dotenv

from googleapiclient import discovery, errors
import logging

load_dotenv(find_dotenv(usecwd=True))

def main():
    parser = argparse.ArgumentParser("Submit a training job to AIP")
    parser.add_argument(
        "--training-inputs",
        type=str,
        help="Path to a YAML file containing the training args dictionary",
        required=True,
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="Unique name of the AIP job",
        required=True
    )
    args = parser.parse_args()
    with open(args.training_inputs, 'r') as training_input_file:
        training_inputs = yaml.safe_load(training_input_file)

    job_spec = {
        'jobId': args.job_name,
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

if __name__ == "__main__":
    main()
