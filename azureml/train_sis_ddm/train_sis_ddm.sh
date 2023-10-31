az ml job create \
    --file azureml/train_sis_ddm/configs/train_sis_ddm_linear.yaml \
    --subscription $AZUREML_SUBSCRIPTION \
    --resource-group $AZUREML_RESSOURCE_GROUP \
    -w $AZUREML_WORKSPACE_NAME --verbose