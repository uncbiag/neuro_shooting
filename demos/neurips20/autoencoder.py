import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.w = w
    
    def forward(self, input):
        nc = input[0].numel()//(self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)

"""
This is essentially the ODE^2VAE encoder/decoder, except that 
we use a shooting block in the latent space. This model is used
when training with a masked loss.
"""
class ShootingAE(nn.Module):
    def __init__(self, n_filt=8, shooting_dim=20, T=16):
        super(ShootingAE, self).__init__()
        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]
        self.shooting_dim = shooting_dim
        self.T = T

        self.encoder = nn.Sequential(
            nn.Conv2d(3, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, shooting_dim)
        self.fc2 = nn.Linear(shooting_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )

        self.blk =  None

    def set_shooting_block(self, sblock):
        self.blk = sblock

    def set_integration_time_vector(self, t):
        assert self.blk is not None
        self.blk.set_integration_time_vector(t, suppress_warning=True)
    
    def forward(self, x):
        # get batch size
        N = x.size(0)
        # forward through encoder
        enc = self.fc1(self.encoder(x))
        # forward through shooting block
        out,_,_,_ = self.blk(enc.unsqueeze(1))
        out = out.squeeze().permute(1,0,2)
        # N x T x shooting_dim
        s = out.contiguous().view(N*self.T,self.shooting_dim)
        # decode
        s = self.decoder(self.fc2(s))
        return out, s.view(N,self.T,1,28,28)  


"""
This is essentially the ODE^2VAE encoder/decoder, except that 
we use a shooting block in the latent space. This model is used
when training by masking the timepoints before the decoder. In 
that way, no excluded timepoints contribute to the batchnorm stats 
during training.
"""
class ShootingAEMasked(nn.Module):
    def __init__(self, n_filt=8, shooting_dim=20, n_skip=4, i_eval=3, T=16):
        super(ShootingAEMasked, self).__init__()

        self.T = T             # timesteps
        self.n_filt = n_filt   # nr. of filters for conv layers
        self.n_skip = n_skip   # how many time points to randomly exclude
        self.i_eval = i_eval   # evaluation timepoint
        self.shooting_dim = shooting_dim
        
        h_dim = n_filt*4**3
        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, self.shooting_dim)
        self.fc2 = nn.Linear(self.shooting_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )

        self.blk =  None

    def set_shooting_block(self, sblock):
        self.blk = sblock

    def set_integration_time_vector(self, t):
        assert self.blk is not None
        self.blk.set_integration_time_vector(t, suppress_warning=True)
    
    def generate_selection_indices(self, batch):
        """
        Generates selection indices for timepoints to 
        be later used for gather(). 
        """
        idx = []
        for _ in range(batch.size(0)):
            valid_idx = [j for j in range(self.T) if j != self.i_eval]       
            np.random.shuffle(valid_idx)
            valid_idx = np.setdiff1d(valid_idx, valid_idx[0:self.n_skip] + [self.i_eval])
            idx.append(torch.tensor([[k]*self.shooting_dim for k in valid_idx]).long().unsqueeze(0))
        return torch.cat(idx)

    def forward(self, x, use_mask):
        """
        Forward gets the data x and, if use_mask=True, masks
        timepoints for decoding. If use_mask=False, the full
        trajectory (i.e., all timepoints) is decoded.
        """
        N = x.size(0)
        # (N x 1 x 28 x 28)
        enc = self.fc1(self.encoder(x)) 
        # (N, shooting_dim)
        out,_,_,_ = self.blk(enc.unsqueeze(1))
        # (T, N, 1, shooting_dim)
        out = out.squeeze().permute(1,0,2)
        # (N, T, shooting_dim)

        if use_mask:
            idx = self.generate_selection_indices(out) # (N, T-n_skip-1, shooting_dim)
            # (N, T, shooting_dim)
            out = out.gather(1, idx)
            # (N, T-n_skip-1, shooting_dim)
            s = out.contiguous().view(N*(self.T-(self.n_skip+1)),self.shooting_dim)
            # (N*(T-n_skip-1), shooting_dim)
            s = self.fc2(s)
            # (N*(T-n_skip-1), hidden_dim)
            s = self.decoder(s)
            # (N*(T-n_skip-1), 1, 28, 28)
            return out, idx, s.view(N,self.T-(self.n_skip+1),1,28,28)   
        else:
            # no need for masking here
            s = out.contiguous().view(N*self.T, self.shooting_dim)
            s = self.fc2(s)
            s = self.decoder(s)
            return out, None, s.view(N,self.T,1,28,28)