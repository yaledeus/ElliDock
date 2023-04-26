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
        loss, (ot_loss, dock_loss, stable_loss, match_loss, rmsd_loss) = self.model(**batch)
        log_type = 'Validation' if val else 'Train'
        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'OT Loss/{log_type}', ot_loss, batch_idx, val)
        self.log(f'Dock Loss/{log_type}', dock_loss, batch_idx, val)
        self.log(f'Stable Loss/{log_type}', stable_loss, batch_idx, val)
        self.log(f'Match Loss/{log_type}', match_loss, batch_idx, val)
        self.log(f'RMSD Loss/{log_type}', rmsd_loss, batch_idx, val)
        return loss
