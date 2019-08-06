import torch
import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block

nonlinearity = 'tanh'
batch_y0 = None

shooting_model = shooting_models.AutoShootingIntegrandModelUpDown(batch_y0=batch_y0, only_random_initialization=True,
                                                                  nonlinearity=nonlinearity)
