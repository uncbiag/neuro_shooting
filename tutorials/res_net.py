import torch
import torch.nn as nn
import torch.nn.functional as F

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block
import neuro_shooting.state_costate_and_data_dictionary_utils as scd_utils
import neuro_shooting.parameter_initialization as parameter_initialization

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
nonlinearity = 'tanh'


class BasicResNet(nn.Module):

    def __init__(self,
                 nr_of_image_channels,
                 layer_channels=[64,128,256,512],
                 nr_of_blocks_per_layer=[3,3,3,3],
                 downsampling_stride=2,
                 nr_of_classes=10
                 ):

        super(BasicResNet, self).__init__()
        self.nr_of_image_channels = nr_of_image_channels
        self.layer_channels=layer_channels
        self.nr_of_blocks_per_layer = nr_of_blocks_per_layer
        self.nr_of_classes = nr_of_classes

        # initial convolution layer
        self.initial_conv = nn.Conv2d(self.nr_of_image_channels, layer_channels[0], kernel_size=3, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(layer_channels[0])

        downsampling_strides = [downsampling_stride]*(len(layer_channels)-1)

        self.shooting_layers = self._create_shooting_layers(layer_channels=self.layer_channels,
                                                            nr_of_blocks_per_layernr_of=self.nr_of_blocks_per_layer_)
        self.striding_layers = self._create_striding_layers(strides=downsampling_strides)

        size_of_last_output_block = None

        self.last_linear = nn.Linear(layer_channels[-1]*size_of_last_output_block, self.nr_of_classes)

    def _create_shooting_layers(self,layer_channels):
        pass

    def _create_striding_layers(self,strides):
        pass

    def forward(self, x):

        # first a convolution, with batch norm and relu
        ret = self.initial_conv(x)
        ret = self.batch_norm(ret)
        ret = F.relu(ret)

        for slayers in self.shooting_layers:
            for b in slayers:
                pass

        CONTINUE HERE


