""" Create Environment

Generate the environment in order to run the training in Azure.

This script should be run from the root directory
>> python environments/create_environment.py

"""

#import required libraries for workspace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
#import required libraries for environments examples
import argparse
import os

SUBSCRIPTION = os.getenv('AZUREML_SUBSCRIPTION')
RESSOURCE_GROUP = os.getenv('AZUREML_RESSOURCE_GROUP')
WORKSPACE = os.getenv('AZUREML_WORKSPACE_NAME')

parser = argparse.ArgumentParser()
parser.add_argument('--environment',
                    help='name of the environment to build...',
                    choices=['diffusers_env'],
                    default="diffusers_env")

if __name__=='__main__':
    args = parser.parse_args()
    #################################
    ### Connect to the workspace
    #################################
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential=credential,subscription_id=SUBSCRIPTION,resource_group_name=RESSOURCE_GROUP,workspace_name=WORKSPACE)
    #################################
    ### Create the docker Context
    #################################
    # We copy sources to env rootdir
    env_root_dir = f"azureml/environments/{args.environment}"
    # We create the env    
    env_docker_context = Environment(
        build=BuildContext(path=env_root_dir,dockerfile_path="Dockerfile"),
        version='1.0.1',
        name=f"{args.environment}",
        description="Environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

