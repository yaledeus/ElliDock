#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import Trainer


class ExpDockTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def get_scheduler(self, optimizer):
        return None

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        out = self.model(**batch)
        log_type = 'Validation' if val else 'Train'
        # TODO: rewrite loss
        # self.log(f'Loss/{log_type}', out.loss, batch_idx, val)
        # return out.loss
        return -1
