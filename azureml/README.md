# Configure Diffusers with AzureML

In order to make this repository work with AzureML, we need some instalations...

# Create env variables.
We have to declare the workspace where we want to work with env variables.
```bash
Add env variable to conda env :
conda env config vars set AZUREML_SUBSCRIPTION='270ae3d6-5d7d-47b8-b82d-17f1eea30f72'
conda env config vars set AZUREML_RESSOURCE_GROUP='AICPDCNEGBL0'
conda env config vars set AZUREML_WORKSPACE_NAME='aicpdevcnemlwweu'
conda env config vars list
```
Login with AzureML.
```bash
pip install azure-cli azure-ai-ml azure-identity
az extension add -n ml
az login
```

