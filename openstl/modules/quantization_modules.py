import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, FSQ, LFQ, RandomProjectionQuantizer, ResidualFSQ, ResidualLFQ
from torch.nn import functional as F


class Quantization(nn.Module):
    def __init__(self, vq_type, codebook_kwargs):
        super(Quantization, self).__init__()

        if vq_type == 'svq':
            self.vq = SVQ(**codebook_kwargs)
        elif vq_type == 'vq':
            self.vq = VectorQuantize(**codebook_kwargs)
        elif vq_type == 'rvq':
            self.vq = ResidualVQ(**codebook_kwargs)
        elif vq_type == 'grvq':
            self.vq = GroupedResidualVQ(**codebook_kwargs)
        elif vq_type == 'mhvq':
            self.vq = VectorQuantize(**codebook_kwargs)
        elif vq_type == 'rfsq':
            self.vq = ResidualFSQ(**codebook_kwargs)
        elif vq_type == 'lfq':
            self.vq = LFQ(**codebook_kwargs)
        elif vq_type == 'rlfq':
            self.vq = ResidualLFQ(**codebook_kwargs)
        else:
            raise Exception('Wrong quantization type!')
            
        self.vq_type = vq_type
        try:
            self.codebook_size = codebook_kwargs['codebook_size']
        except:
            pass
    
    def forward(self, embed):
        # embed: BT, C, H, W
        auxilary_loss = dict()
        if self.vq_type == 'svq':
            embed_vq, code_weight, codebook, perplexity = self.vq(embed)
            return embed_vq, auxilary_loss, perplexity, code_weight, codebook
            
        perplexity = dict()
        x_shape = embed.shape
        B, C, H, W = embed.shape
        x_flat = embed.view(embed.size(0), embed.size(1), -1).transpose(1,2)
        
        if self.vq_type in ['vq', 'mhvq', 'rvq', 'grvq']:
            x_flat, indices, loss = self.vq(x_flat)
            loss = torch.mean(loss)
            if self.training:
                if len(indices.shape) == 2:
                    # torch.Size([12, 512])
                    perplexity['perp'] = self.evaluate_codebook_usage(indices.unsqueeze(0))
                elif len(indices.shape) == 3:
                    # torch.Size([12, 512, 8])  BT, HW, head
                    perplexity['perp'] = self.evaluate_codebook_usage(indices.permute(2,0,1))
                elif len(indices.shape) == 4:
                    # torch.Size([2, 12, 512, 8]) Group, BT, HW, head
                    indices = indices.permute(0,3,1,2)
                    perplexity['perp'] = self.evaluate_codebook_usage(indices.reshape(-1,indices.shape[2],indices.shape[3]))
            if self.vq_type in ['vq', 'mhvq']:
                codebook = self.vq._codebook.embed
            elif self.vq_type in ['rvq', 'grvq']:
                codebook = self.vq.codebooks
        else:
            if self.vq_type == 'rfsq':
                x_flat, indices = self.vq(x_flat)
                loss = torch.tensor([0.], device = x_flat.device, requires_grad = self.training)
            elif self.vq_type == 'lfq':
                x_flat, indices, loss = self.vq(embed)
                x_flat = x_flat.view(embed.size(0), embed.size(1), -1).transpose(1,2)
            elif self.vq_type == 'rlfq':
                x_flat, indices, loss = self.vq(x_flat)
                loss = torch.mean(loss)
            codebook = torch.tensor([0.], device = x_flat.device, requires_grad = self.training)
            
        # print(codebook.std(-1))
        auxilary_loss['loss_vq'] = loss
        code_weight = indices
        x_flat = x_flat.transpose(1,2)
        embed_vq = x_flat.view(*x_shape)
        return embed_vq, auxilary_loss, perplexity, code_weight, codebook

        
    def evaluate_codebook_usage(self, indices):
        # # (group) * head, BT, HW
        # dtype = indices.dtype
        # H, L, T = indices.shape
        # avg_probs = F.one_hot(indices.reshape(H, L*T), num_classes=self.codebook_size).type(dtype) # H, LT, codebook size
        # avg_probs = avg_probs.sum(1) / avg_probs.shape[1]
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), axis=-1)).mean()
        # return perplexity
        return torch.tensor([0.], device = indices.device, requires_grad = self.training)
    
    
class SVQ(nn.Module):
    def __init__(self,
                 input_dim: int,
                 codebook_size: int,
                 freeze_codebook: bool,
                 nonlinear: bool,
                 middim: int,
                 init_method='normal',
                 evaluate_usage=False,
                 hid_relu=True,
                 out_relu=False
                ):
        super(SVQ, self).__init__()
        self.proj_regression_weight = MLPProjection(input_dim, codebook_size, middim, codebook_size, nonlinear, hid_relu, out_relu)

        codebook = torch.empty(codebook_size, input_dim)
        if init_method == 'kaiming':
            nn.init.kaiming_uniform_(codebook)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(codebook, gain=nn.init.calculate_gain('relu'))
        elif init_method == 'sparse':
            nn.init.sparse_(codebook, sparsity=0.9)
        elif init_method == 'trunc':
            nn.init.trunc_normal_(codebook)
        elif init_method == 'orthogonal':
            nn.init.orthogonal_(codebook)
        elif init_method == 'random':
            codebook = torch.randn(codebook_size, input_dim)
        elif init_method == 'normal':
            nn.init.normal_(codebook)
        else:
            raise Exception('Wrong initialize method!')
        self.codebook = nn.Parameter(codebook, requires_grad=not freeze_codebook)
        self.codebook_size = codebook_size
        self.evaluate_usage = evaluate_usage
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(B, H*W, C)
        x = self.proj_regression_weight(x)
        if self.training and self.evaluate_usage:
            perplexity = self.evaluate_codebook_usage(x)
        else:
            perplexity = dict()
        sparsereg = torch.bmm(x, self.codebook.unsqueeze(0).expand(B, *self.codebook.shape))
        sparsereg = sparsereg.reshape(B, H, W, C).permute(0,3,1,2)
        return sparsereg, x, self.codebook, perplexity

    def evaluate_codebook_usage(self, x):
        dtype = x.dtype
        perplexity_all = dict()
        x_norm = (x-x.mean()) / x.std()
        x_norm = x_norm.abs()
        perplexity_all['perp_3'] = self.calculate_perplexity((x_norm>3).type(dtype))
        perplexity_all['perp_2'] = self.calculate_perplexity((x_norm>2).type(dtype))
        return perplexity_all

    def calculate_perplexity(self, indices):
        # 12, 512, 1024   T, HW, codebook size
        t, l, s = indices.shape
        indices = indices.reshape(-1, s)
        avg_probs = indices.sum(0) / indices.sum()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), axis=-1))
        perplexity = perplexity.mean()
        return perplexity
    
    
class MLPProjection(nn.Module):
    def __init__(self, N, M, k, codebook_size, nonlinear, hid_relu, out_relu):
        super(MLPProjection, self).__init__()
        if nonlinear:
            if codebook_size < k: # Limit the complexity if codebook size is smaller than hidden dimension(k)
                k = codebook_size
            self.linear_proj = nn.Linear(in_features=N, out_features=k, bias=True)
            self.linear_proj2 = nn.Linear(in_features=k, out_features=M, bias=True)
            self.activation_mid = nn.ReLU() if hid_relu else nn.Identity()
            self.activation_out = nn.ReLU() if out_relu else nn.Identity()
        else:
            self.linear_proj = nn.Linear(in_features=N, out_features=M, bias=True)
        self.nonlinear = nonlinear
        
    def forward(self, x):
        if self.nonlinear:
            x = self.linear_proj(x)
            x = self.activation_mid(x)
            x = self.linear_proj2(x)
            x = self.activation_out(x)
        else:
            x = self.linear_proj(x)
        return x
    