#!/usr/bin/env python3

"""Fits a linear aesthetic model to precomputed CLIP embeddings."""

import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=str, help='the input feature vectors')
    p.add_argument('output', type=str, help='the output model')
    p.add_argument('--val-size', type=float, default=0.1, help='the validation set size')
    p.add_argument('--seed', type=int, default=0, help='the random seed')
    args = p.parse_args()

    train_set = torch.load(args.input, map_location='cpu')
    X = F.normalize(train_set['embeds'].float(), dim=-1).numpy()
    X *= X.shape[-1] ** 0.5
    y = train_set['ratings'].numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.seed)
    regression = Ridge()
    regression.fit(X_train, y_train)
    score_train = regression.score(X_train, y_train)
    score_val = regression.score(X_val, y_val)
    print(f'Score on train: {score_train:g}')
    print(f'Score on val: {score_val:g}')
    model = AestheticMeanPredictionLinearModel(X_train.shape[1])
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(regression.coef_))
        model.linear.bias.copy_(torch.tensor(regression.intercept_))
    torch.save(model.state_dict(), args.output)


if __name__ == '__main__':
    main()
