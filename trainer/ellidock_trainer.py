#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import Trainer


class ElliDockTrainer(Trainer):

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
        loss, (fit_loss, overlap_loss, ref_loss, dock_loss, stable_loss, rmsd_loss) = self.model(**batch)
        log_type = 'Validation' if val else 'Train'
        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'Fit Loss/{log_type}', fit_loss, batch_idx, val)
        self.log(f'Overlap Loss/{log_type}', overlap_loss, batch_idx, val)
        self.log(f'Ref Loss/{log_type}', ref_loss, batch_idx, val)
        self.log(f'Dock Loss/{log_type}', dock_loss, batch_idx, val)
        self.log(f'Stable Loss/{log_type}', stable_loss, batch_idx, val)
        self.log(f'RMSD Loss/{log_type}', rmsd_loss, batch_idx, val)
        return loss
