"""
HGIMamba
"""
import torch
import torch.nn as nn
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
import torch.nn.functional as F
import math
import numpy as np
import time
from einops import rearrange, repeat
from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba.mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Fuse_SSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, z=None, inference_params=None):

        batch, seqlen, dim = hidden_states.shape

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x = rearrange(hidden_states, "b l d -> b d l", l=seqlen)
        z = None if z is None else rearrange(z, "b l d -> b d l", l=seqlen)

        x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        y = rearrange(y, "b d l -> b l d")

        return self.out_proj(y)

class IFBlock(nn.Module):
    def __init__(self, in_dim, d_state, expand, bias=False, conv_bias=True, d_conv=4):
        super(IFBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(in_dim)
        self.ln_2 = nn.LayerNorm(in_dim)

        self.expand = expand

        self.d_inner = int(self.expand * in_dim)
        
        self.in_proj1 = nn.Linear(in_dim, self.d_inner * 2, bias=bias)
        self.in_proj2 = nn.Linear(in_dim, self.d_inner * 2, bias=bias)

        self.act = nn.SiLU()
        
        self.attention1 = Fuse_SSM(d_model=in_dim, d_state=d_state, expand=expand)
        self.attention2 = Fuse_SSM(d_model=in_dim, d_state=d_state, expand=expand)

        
        
    def forward(self, x_he, x_ihc):
        he_ = self.ln_1(x_he)  # [B, l, 256]
        ihc_ = self.ln_2(x_ihc)  # [B, l, 256]

        he_12 = self.in_proj1(he_)  # [B, l, 1024]
        he_1, he_2 = he_12.chunk(2, dim=-1)  # [B, l, 512]

        ihc_12 = self.in_proj2(ihc_)  # [B, l, 1024]
        ihc_1, ihc_2 = ihc_12.chunk(2, dim=-1)  # [B, l, 512]

        z_he = self.act(he_2)          # [B, l, 512]
        z_ihc = self.act(ihc_2)        # [B, l, 512]
        
        # he_out = self.attention1(hidden_states=he_1, z=z_he)  # [B, L, 512]
        he_out = self.attention1(hidden_states=he_1, z=z_ihc)  # [B, L, 512]
        ihc_out = self.attention2(hidden_states=ihc_1, z=z_he)  # [B, L, 512]

        # h = he_out + ihc_out  # [B, l, 512]

        # h = self.out_proj(h)  # [B, l, 256]

        he = he_out + x_he       # [B, l, 256]
        ihc = ihc_out + x_ihc     # [B, l, 256]

        return he, ihc
    
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        # x = x.unsqueeze(0)
        ## x: N x L
        # x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights( A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N

class SingleMambaBlock(nn.Module):
    def __init__(self, mamba_type='Mamba', in_dim=1024, d_state=16):
        super(SingleMambaBlock, self).__init__()
        if mamba_type == 'Mamba':
            self.encoder = Mamba(d_model=in_dim, d_state=d_state, d_conv=4, expand=2)
        elif mamba_type == 'BiMamba':
            self.encoder = BiMamba(d_model=in_dim, d_state=d_state, d_conv=4, expand=2)
        else :
            self.encoder = SRMamba(d_model=in_dim, d_state=d_state, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        x_ = self.norm(x)
        output = self.encoder(x_) + x
        return output

class HGPMamba(nn.Module):
    def __init__(self, in_dim, d_state=16, n_classes=4, dropout=0.1, act='relu', mamba_type='Mamba', fusion=None):
        super(HGPMamba, self).__init__()
        
        ### FC Layer over WSI bag
        fc = [nn.Linear(in_dim, 256), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)
        
        self.mlp = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU()
            )

        self.norm = nn.LayerNorm(256)
        
        self.n_classes = n_classes

        if fusion == 'IFBlock':
            self.fusion1 = IFBlock(in_dim=256, d_state=d_state, expand=2)
            self.fusion2 = IFBlock(in_dim=256, d_state=d_state, expand=2)
            self.mm = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU()])
        elif fusion == 'concat':
            self.attention_path = Attention_Gated(L=256, D=128, K=1)
            self.attention_omic = Attention_Gated(L=256, D=128, K=1)
            self.mm = nn.Sequential(*[nn.Linear(256*2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])
        else:
            self.mm = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU()])
            print('Please specify the fusion method!')
        
        self.multi_enhancement = nn.ModuleList([SingleMambaBlock(mamba_type, 256, d_state) for _ in range(1)])
        self.classifier = nn.Linear(256, n_classes)

        self.fusion = fusion
        
        self.apply(initialize_weights)


    def forward(self, x, y):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        x = x.float()  
        f_WSI = self.wsi_net(x)

        if len(y.shape) == 2:
            y = y.expand(1, -1, -1)
        y = y.float() 
        f_mIF = self.mlp(y)

        if self.fusion == 'concat':
            AA_path = self.attention_path(f_WSI)
            h_path_bag = torch.mm(AA_path, f_WSI)
            AA_omic = self.attention_omic(f_mIF)
            h_omic_bag = torch.mm(AA_omic, f_mIF)
            h = torch.cat([h_path_bag, h_omic_bag], dim=-1) 
            fea = self.mm(h)  
        elif self.fusion == 'IFBlock':
            f_WSI, f_mIF= self.fusion1(f_WSI, f_mIF)     #[B, L, D]
            f_WSI, f_mIF= self.fusion2(f_WSI, f_mIF)     #[B, L, D]

            h = torch.cat([f_WSI, f_mIF], dim=1) 
            for layer in self.multi_enhancement:
                h = layer(h)             
            fea = self.mm(h)
        elif self.fusion is None:
            h = torch.cat([f_WSI, f_mIF], dim=1) 
            for layer in self.multi_enhancement:
                h = layer(h)
            fea = self.mm(h)
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        fea = self.norm(fea)
        fea = torch.max(fea, dim=1)[0]  # [B, 256]

        logits = self.classifier(fea)
        # Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat, None, None
        # return logits
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wsi_net = self.wsi_net.to(device)
        self.mlp = self.mlp.to(device)

        self.fusion1 = self.fusion1.to(device)
        self.fusion2 = self.fusion2.to(device)
        self.multi_enhancement = self.multi_enhancement.to(device)

        self.norm = self.norm.to(device)
        self.mm = self.mm.to(device)
        self.classifier = self.classifier.to(device)


if __name__ == '__main__':
    data_h = torch.randn((6000, 512))
    data_p = torch.randn((6000, 50))
    data_h, data_p = data_h.cuda(), data_p.cuda()
    model = HGPMamba(fusion='concat', mamba_type='BiMamba', n_classes=4, dropout=0.25, in_dim=512).cuda()
    # total_params = sum([param.numel() for param in model.parameters()])
    print(model.eval())
    # flops, params = profile(model, inputs=(data_h, data_p))
    start_time = time.time()
    out = model(x=data_h, y=data_p)
    end_time = time.time()
    infer_time = end_time - start_time
    memory_allocated = torch.cuda.max_memory_allocated()
    Gb_consumed = memory_allocated / 1e9
    print('Inference time:          %.4f +/- %.4f s\n' % (np.mean(infer_time), np.std(infer_time)))
    print('Memory occupation:       %.4f Gb\n' % Gb_consumed)
    # print(f"THOP: FLOPs: {flops/1e9:.3f} G, Params: {params/1e6:.3f} M")