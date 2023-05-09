import torch
from torch import nn
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel

class MultilingualCLIP(XLMRobertaPreTrainedModel):
    def __init__(self, config, in_features=1024, out_features=768): # 1024, 768
        super().__init__(config)
        self.transformer = XLMRobertaModel(config)
        self.LinearTransformation = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
            dim=1
        )[:, None]
        return self.LinearTransformation(embs2), embs
