import torch
import torch.nn as nn
import torch.nn.functional as F

import neuro_shooting.shooting_blocks as shooting_blocks
import neuro_shooting.shooting_models as shooting_models
import neuro_shooting.striding_block as striding_block
import neuro_shooting.parameter_initialization as parameter_initialization
import neuro_shooting.activation_functions_and_derivatives as ad

gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

class BasicResNet(nn.Module):

    def __init__(self,
                 nr_of_image_channels,
                 layer_channels=[64,128,256,512],
                 nr_of_blocks_per_layer=[3,3,3,3],
                 downsampling_stride=2,
                 nonlinearity='tanh',
                 particle_sizes=[[15,15],[11,11],[7,7],[5,5]],
                 nr_of_particles=10,
                 nr_of_classes=10,
                 parameter_weight=1.0,
                 inflation_factor = 2,
                 optimize_over_data_initial_conditions=False,
                 optimize_over_data_initial_conditions_type="linear"
                 ):

        super(BasicResNet, self).__init__()
        self.nr_of_image_channels = nr_of_image_channels
        self.layer_channels=layer_channels
        self.nr_of_blocks_per_layer = nr_of_blocks_per_layer
        self.nr_of_classes = nr_of_classes
        self.pw = parameter_weight
        self.nonlinearity = nonlinearity
        self.inflation_factor = inflation_factor
        self.nl,_ = ad.get_nonlinearity(nonlinearity=nonlinearity)
        self.optimize_over_data_initial_conditions = optimize_over_data_initial_conditions
        self.optimize_over_data_initial_conditions_type = optimize_over_data_initial_conditions_type
        self._state_initializer =  parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.5)
        self._costate_initializer = parameter_initialization.ConvolutionEvolutionParameterInitializer(only_random_initialization=True, random_initialization_magnitude=0.0)

        self.nr_of_particles = nr_of_particles
        self.particle_sizes = particle_sizes

        if len(layer_channels)!=len(self.particle_sizes):
            raise ValueError('Dimension mismatch, between laters and particle sizes. A particle size needs to be defined for each layer.')

        # initial convolution layer
        self.initial_conv = nn.Conv2d(self.nr_of_image_channels, layer_channels[0], kernel_size=1, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(layer_channels[0])
        self.conv_res = nn.Conv2d(layer_channels[0],layer_channels[0],kernel_size=3, stride=1, padding=1, bias=False)
        downsampling_strides = [downsampling_stride]*(len(layer_channels)-1) + [None]

        #self.shooting_model = shooting_models.AutoShootingIntegrandModelConv2DBatch
        #self.shooting_model = shooting_models.AutoShootingIntegrandModelSimpleConv2D
        self.shooting_model = shooting_models.AutoShootingIntegrandModelUpDownConv2D
        #self.shooting_model = shooting_models.AutoShootingOptimalTransportSimpleConv2D
        self.shooting_layers = self._create_shooting_layers(layer_channels=self.layer_channels,
                                                            nr_of_blocks_per_layer=self.nr_of_blocks_per_layer)

        self.striding_blocks = self._create_striding_blocks(strides=downsampling_strides)

        self.last_linear = nn.Linear(layer_channels[-1], self.nr_of_classes)

        self._forward_not_yet_executed = True

    def parameters(self, recurse=True):
        if self._forward_not_yet_executed:
            raise ValueError('Parameters are created dynamically. Please execute your entire pipeline once first before calling parameters()!')
        # we overwrite this to assure that one forward pass has been done (so we can collect parameters)
        return super(BasicResNet, self).parameters(recurse=recurse)


    def _create_one_shooting_block(self,name, nr_of_channels,nr_of_particles,particle_size, particle_dimension, only_pass_through=False):
        if only_pass_through:
            shooting_model = self.shooting_model(in_features=nr_of_channels,
                                                 nonlinearity=self.nonlinearity,
                                                 nr_of_particles=None)
        else:
            shooting_model = self.shooting_model(in_features=nr_of_channels,
                                                 nonlinearity=self.nonlinearity,
                                                 state_initializer=self._state_initializer,
                                                 costate_initializer=self._costate_initializer,
                                                 nr_of_particles=nr_of_particles,
                                                 particle_size=particle_size,
                                                 particle_dimension=particle_dimension,parameter_weight=self.pw,inflation_factor = self.inflation_factor,optimize_over_data_initial_conditions=self.optimize_over_data_initial_conditions,
            optimize_over_data_initial_conditions_type=self.optimize_over_data_initial_conditions_type)

        shooting_block = shooting_blocks.ShootingBlockBase(name=name, shooting_integrand=shooting_model,parameter_weight = 0.01)

        return shooting_block

    def _compute_nr_of_additional_channels(self,layer_channels):

        nr_of_additional_channels = []
        for i,nr_of_channels in enumerate(layer_channels):
            if i==0:
                nr_of_additional_channels.append(nr_of_channels)
            else:
                nr_of_additional_channels.append(nr_of_channels-layer_channels[i-1])
        return nr_of_additional_channels

    def _create_shooting_layers(self,layer_channels,nr_of_blocks_per_layer):
        layer_base_name = 'shooting_layer'
        shooting_layers = []

        nr_of_additional_channels = self._compute_nr_of_additional_channels(layer_channels)
        for layer_nr,nr_of_channels in enumerate(layer_channels):
            shooting_blocks_for_layer = []
            for shooting_block in range(nr_of_blocks_per_layer[layer_nr]):
                current_name = layer_base_name + '_' + str(layer_nr) + '_' + str(shooting_block)

                only_pass_through = shooting_block>0
                current_block = self._create_one_shooting_block(name=current_name,
                                                                nr_of_channels=nr_of_channels,
                                                                nr_of_particles=self.nr_of_particles,
                                                                particle_size=self.particle_sizes[layer_nr],
                                                                particle_dimension=nr_of_additional_channels[layer_nr],
                                                                only_pass_through=only_pass_through)
                shooting_blocks_for_layer.append(current_block)
                # and also register it with the module
                self.add_module(name=current_name,module=current_block)

            shooting_layers.append(shooting_blocks_for_layer)

        return shooting_layers



    def _create_striding_blocks(self,strides):

        striding_blocks = []
        for stride in strides:
            if stride is None:
                striding_blocks.append(None)
            else:
                current_striding_block = striding_block.ShootingStridingBlock(stride=[stride]*2,stride_dims=[2,3])
                striding_blocks.append(current_striding_block)
        return striding_blocks

    def forward(self, x):

        # first a convolution, with batch norm and relu

        ret = self.initial_conv(x)
        #ret = x
        #ret = self.batch_norm(ret)
        temp = ret
        ret = self.nl(ret)

        ret = self.conv_res(ret) + temp
        is_first = True

        for shooting_layer,striding_block in zip(self.shooting_layers,self.striding_blocks):
            for shooting_block in shooting_layer:
                if is_first:
                    _, state_dicts, costate_dicts, data_dicts = shooting_block(x=ret)
                    is_first = False
                else:
                    ret, state_dicts, costate_dicts, data_dicts = shooting_block(data_dict_of_dicts=data_dicts,
                                                               pass_through_state_dict_of_dicts=state_dicts,
                                                               pass_through_costate_dict_of_dicts=costate_dicts)

            #we do striding after each shooting layer
            if striding_block is not None:
                state_dicts, costate_dicts, data_dicts = striding_block(state_dict_of_dicts=state_dicts,
                                                                        costate_dict_of_dicts=costate_dicts,
                                                                        data_dict_of_dicts=data_dicts)

        # now we can apply some global average pooling if desired and then apply the linear layer
        #ret = F.avg_pool2d(ret,4)
        ret = nn.AdaptiveMaxPool2d(1)(ret)
        ret = ret.view(ret.size(0), -1)
        ret = self.last_linear(ret)

        self._forward_not_yet_executed = False

        return ret

# batch_size = 5
# sample_cifar_data = torch.zeros(batch_size,3,32,32)
#
# # todo: if one only uses one layer the return data is not properly done, but returns a SortedDict. Needs to be fixed within shooting block.
#
# res_net = BasicResNet(nr_of_image_channels=3,nr_of_blocks_per_layer=[2,2,2,2])
# ret = res_net(sample_cifar_data)


