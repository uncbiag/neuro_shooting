import torch
import torch.nn as nn
from abc import abstractmethod

class ParameterInitializer(object):

    def __init__(self,only_random_initialization=True,random_initialization_magnitude=0.5,sample_batch=None):
        super(ParameterInitializer, self).__init__()
        self.sample_batch = sample_batch
        self.only_random_initialization=only_random_initialization
        self.random_initialization_magnitude = random_initialization_magnitude

    @abstractmethod
    def create_zero_parameters(self,nr_of_particles,particle_dimension,particle_size,*argv,**kwargs):
        pass

    @abstractmethod
    def create_random_parameters(self, nr_of_particles, particle_dimension, particle_size, *argv, **kwargs):
        pass

    def create_custom_parameters(self, nr_of_particles, particle_dimension, particle_size, *argv, **kwargs):
        raise ValueError('Not implemented. If you want to use this functionalty, derive an appropriate class.')


    def create_zero_parameters_like(self, like_tensor, *argv, **kwargs):
        return nn.Parameter(torch.zeros_like(like_tensor))

    def create_random_parameters_like(self, like_tensor, *argv, **kwargs):
        return nn.Parameter(self.random_initialization_magnitude * torch.randn_like(like_tensor))

    def create_custom_parameters_like(self, like_tensor, *argv, **kwargs):
        raise ValueError('Not implemented. If you want to use this functionalty, derive an appropriate class.')

    def create_parameters_like(self,like_tensor,set_to_zero=False,*argv,**kwargs):

        if set_to_zero:
            return self.create_zero_parameters_like(like_tensor=like_tensor,*argv,**kwargs)
        elif self.only_random_initialization:
            return self.create_random_parameters_like(like_tensor=like_tensor,*argv,**kwargs)
        else:
            return self.create_custom_parameters_like(like_tensor=like_tensor,*argv,**kwargs)

    def create_zero_parameters_of_size(self, size, *argv, **kwargs):
        return nn.Parameter(torch.zeros(size))

    def create_random_parameters_of_size(self, size, *argv, **kwargs):
        return nn.Parameter(self.random_initialization_magnitude * torch.randn(size))

    def create_custom_parameters_of_size(self, size, *argv, **kwargs):
        raise ValueError('Not implemented. If you want to use this functionalty, derive an appropriate class.')

    def create_parameters_of_size(self,size,set_to_zero=False,*argv,**kwargs):

        if set_to_zero:
            return self.create_zero_parameters_of_size(size=size, *argv, **kwargs)
        elif self.only_random_initialization:
            return self.create_random_parameters_of_size(size=size, *argv, **kwargs)
        else:
            return self.create_custom_parameters_of_size(size=size, *argv, **kwargs)

    def create_parameters(self,nr_of_particles,particle_size,particle_dimension=1,set_to_zero=False,*argv,**kwargs):

        if set_to_zero:
            return self.create_zero_parameters(nr_of_particles=nr_of_particles,
                                               particle_size=particle_size,
                                               particle_dimension=particle_dimension,
                                               *argv,**kwargs)
        elif self.only_random_initialization:
            return self.create_random_parameters(nr_of_particles=nr_of_particles,
                                                 particle_size=particle_size,
                                                 particle_dimension=particle_dimension,
                                                 *argv,**kwargs)
        else:
            return self.create_custom_parameters(nr_of_particles=nr_of_particles,
                                                 particle_size=particle_size,
                                                 particle_dimension=particle_dimension,
                                                 *argv,**kwargs)


class VectorEvolutionParameterInitializerBase(ParameterInitializer):
    def __init__(self,only_random_initialization=True,random_initialization_magnitude=0.5,sample_batch=None):
        super(VectorEvolutionParameterInitializerBase, self).__init__(only_random_initialization=only_random_initialization,
                                                                  random_initialization_magnitude=random_initialization_magnitude,
                                                                  sample_batch=sample_batch)

    def create_zero_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):
        size = tuple([nr_of_particles,particle_dimension,particle_size])
        return nn.Parameter(torch.zeros(size))

class VectorEvolutionParameterInitializer(VectorEvolutionParameterInitializerBase):
    def __init__(self, only_random_initialization=True, random_initialization_magnitude=0.5, sample_batch=None):
        super(VectorEvolutionParameterInitializer, self).__init__(
            only_random_initialization=only_random_initialization,
            random_initialization_magnitude=random_initialization_magnitude,
            sample_batch=sample_batch)

    def create_random_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):
        size = tuple([nr_of_particles,particle_dimension,particle_size])
        return nn.Parameter(self.random_initialization_magnitude*torch.randn(size))

class VectorEvolutionSampleBatchParameterInitializer(VectorEvolutionParameterInitializerBase):
    def __init__(self, only_random_initialization=True, random_initialization_magnitude=0.5, sample_batch=None):
        super(VectorEvolutionSampleBatchParameterInitializer, self).__init__(
            only_random_initialization=only_random_initialization,
            random_initialization_magnitude=random_initialization_magnitude,
            sample_batch=sample_batch)

    def create_random_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):
        size = tuple([nr_of_particles,particle_dimension,particle_size])
        return nn.Parameter(self.sample_batch + self.random_initialization_magnitude*torch.randn(size))


class ConvolutionEvolutionParameterInitializerBase(ParameterInitializer):
    def __init__(self, only_random_initialization=True, random_initialization_magnitude=0.5,sample_batch=None):
        super(ConvolutionEvolutionParameterInitializerBase, self).__init__(only_random_initialization=only_random_initialization,
                                                                           random_initialization_magnitude=random_initialization_magnitude,
                                                                           sample_batch=sample_batch)

    def create_zero_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):

        if type(particle_size)!=tuple and type(particle_size)!=list:
            raise ValueError('Expected the particle size as a tuple or list e.g., [3,3], but got {}.'.format(type(particle_size)))

        size = tuple([nr_of_particles,particle_dimension,*particle_size])
        return nn.Parameter(torch.zeros(size))

class ConvolutionEvolutionParameterInitializer(ConvolutionEvolutionParameterInitializerBase):
    def __init__(self, only_random_initialization=True, random_initialization_magnitude=0.5, sample_batch=None):
        super(ConvolutionEvolutionParameterInitializer, self).__init__(
            only_random_initialization=only_random_initialization,
            random_initialization_magnitude=random_initialization_magnitude,
            sample_batch=sample_batch)

    def create_random_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):
        if type(particle_size) != tuple and type(particle_size) != list:
            raise ValueError('Expected the particle size as a tuple or list e.g., [3,3], but got {}.'.format(type(particle_size)))

        size = tuple([nr_of_particles,particle_dimension,*particle_size])
        return nn.Parameter(self.random_initialization_magnitude*torch.randn(size))
        return nn.Parameter(self.random_initialization_magnitude*torch.randn(size))

class ConvolutionEvolutionSampleBatchParameterInitializer(ConvolutionEvolutionParameterInitializerBase):
    def __init__(self, only_random_initialization=True, random_initialization_magnitude=0.5, sample_batch=None):
        super(ConvolutionEvolutionSampleBatchParameterInitializer, self).__init__(
            only_random_initialization=only_random_initialization,
            random_initialization_magnitude=random_initialization_magnitude,
            sample_batch=sample_batch)

    def create_random_parameters(self,nr_of_particles,particle_size,particle_dimension=1,*argv,**kwargs):
        if type(particle_size) != tuple and type(particle_size) != list:
            raise ValueError('Expected the particle size as a tuple or list e.g., [3,3], but got {}.'.format(type(particle_size)))

        size = tuple([nr_of_particles,particle_dimension,*particle_size])
        return nn.Parameter(self.sample_batch + self.random_initialization_magnitude*torch.randn(size))
        return nn.Parameter(self.sample_batch + self.random_initialization_magnitude*torch.randn(size))