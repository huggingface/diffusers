# CREATE 256
az ml job create \
    --file azureml/train_vae_ldm/configs/train_vae_ldm_256.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose

# CREATE 512
az ml job create \
    --file azureml/train_vae_ldm/configs/train_vae_ldm_512.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose