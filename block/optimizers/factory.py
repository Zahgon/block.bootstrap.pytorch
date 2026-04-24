import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.optimizers.factory import factory_optimizer
from .lr_scheduler import ReduceLROnPlateau
from .lr_scheduler import BanOptimizer

def factory(model, engine):
    pass
