az ml job create \
    --file azureml/train_sis_ldm/configs/train_sis_ldm_256.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose