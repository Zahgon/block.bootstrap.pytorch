import torch
import torch.nn as nn
from bootstrap.lib.logger import Logger

class ReduceLROnPlateau():

    def __init__(self,
            optimizer,
            engine=None,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=False,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08):
        raise NotImplementedError
            #engine.register_hook('eval_on_end_epoch', self.step_lr_scheduler)

    def step_lr_scheduler(self):
        pass

    def __getattr__(self, key):
        raise NotImplementedError


# Inspired from https://github.com/jnhwkim/ban-vqa/blob/master/train.py
class BanOptimizer():

    def __init__(self, engine,
            name='Adamax',
            lr=0.0007,
            gradual_warmup_steps=[0.5, 2.0, 4],
            lr_decay_epochs=[10, 20, 2],
            lr_decay_rate=.25):
        raise NotImplementedError

    def set_lr(self):
        pass

    def display_norm(self):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass

    def __getattr__(self, key):
        raise NotImplementedError
