import torch
import torch.nn as nn
from einops import rearrange,repeat,reduce
'''
以上的三个分别用于维度改变、图像复制、维度削减
https://www.cnblogs.com/c-chenbin/p/15375637.html
'''
from Transformer_Explainability.modules.layers_ours import *
from Transformer_Explainability.modules.layers_ours import DWConvolution,DWConToImage

# from ..DPvT.helpers import load_pretrained
# from timm.models.layers import trunc_normal_
from Transformer_Explainability.baselines.DPvT.layer_helpers import to_2tuple,trunc_normal_
#**kwargs 可变变量-字典  *args 元组
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }
default_cfgs = {
    #patch models
    'DPvT-b0' : _cfg(
        url = '/home/zhangfeng/huangshixin/Btrain/A_single_car/output/model-1-MT-b0/ckpt_epoch_299.pth'
    ),
    'DPvT-b1' : _cfg(
        url='/home/zhangfeng/huangshixin/Btrain/A_single_car/output/model-1-MT-b0/ckpt_epoch_299.pth',
        mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5),
    )
}
#输入的这个all_layer_matrices  ---指代所有层的评估值
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

#系统的模型

import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
import torch.nn.functional as F
# from torch import einsum
'''
可视化时候 并不需要torch附带实际的功能
'''
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.dwconv = DWConvolution(hidden_features)#只需要输入dim即可
        self.act = GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = ReLU(inplace=True)

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        #改动 dwc加上一个跳跃分枝再进行训练  dwc之后会生成一个序列
        x = self.dwconv(x, H, W) + x
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def relprop(self,cam,**kwargs):

        cam = self.drop.relprop(cam,**kwargs)
        cam = self.fc2.relprop(cam,**kwargs)
        cam = self.drop.relprop(cam,**kwargs)
        cam = self.act.relprop(cam,**kwargs)
        cam = self.dwconv.relprop(cam,**kwargs)
        if self.linear:
            cam = self.relu.relprop(cam,**kwargs)
            return self.fc1.relprop(cam,**kwargs)
        else:
            return self.fc1.relprop(cam,**kwargs)
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

    def relprop(self,cam,**kwargs):
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.shape[0], cam.shape[1],
                          (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        cam = self.norm.relprop(cam,**kwargs)
        return self.proj.relprop(cam, **kwargs)

# class Attention_PVTV2(nn.Module):
#
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
#
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # A = Q*K^T
#         self.matmul1 = einsum('bhid,bhjd->bhij')
#         # attn = A*V
#         self.matmul2 = einsum('bhij,bhjd->bhid')
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.linear = linear
#         self.sr_ratio = sr_ratio
#         if not linear:
#             if sr_ratio > 1:
#                 self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#                 self.norm = nn.LayerNorm(dim)
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(7)
#             self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#             self.norm = nn.LayerNorm(dim)
#             self.act = nn.GELU()
#         self.apply(self._init_weights)#自定义参数初始化方式
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):#是否是一个实例
#             trunc_normal_(m.weight, std=.02)#正太分布
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)#初始化值
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def get_attn(self):
#         return self.attn
#     def save_attn(self, attn):
#         self.attn = attn
#     def save_attn_cam(self, cam):
#         self.attn_cam = cam
#     def get_attn_cam(self):
#         return self.attn_cam
#     def get_v(self):
#         return self.v
#     def save_v(self, v):
#         self.v = v
#     def save_v_cam(self, cam):
#         self.v_cam = cam
#     def get_v_cam(self):
#         return self.v_cam
#     def save_attn_gradients(self, attn_gradients):
#         self.attn_gradients = attn_gradients
#     def get_attn_gradients(self):
#         return self.attn_gradients
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#permute 数据的位置
#
#         if not self.linear:
#             if self.sr_ratio > 1:
#                 x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#                 x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#                 x_ = self.norm(x_)
#                 kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             else:
#                 kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             x_ = self.act(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         # print(f"-------------attn:{(attn@v).shape}")
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#     def relprop(self,cam,**kwargs):
#         b,n,c  = cam.shape
#         cam = self.proj_drop.relprop(cam,**kwargs)
#         cam = self.proj.relprop(cam,**kwargs)
#         # cam = rearrange(cam, 'b n c -> b h n d', h=self.num_heads)
#         cam = rearrange(cam,'b n c -> b c h w',h=int(math.sqrt(n)),w=int(math.sqrt(n)))
#
#         #处理矩阵相乘
#         # attn = A*V
#         (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
#         cam1 /= 2
#         cam_v /= 2
#         self.save_v_cam(cam_v)
#         self.save_attn_cam(cam1)
#
#         cam1 = self.attn_drop(cam_v)
#         #在softmax中进行注册
#         cam1 = cam1.softmax.relprop(cam1, **kwargs)
#
#         #计算A = Q*K^t
#         (cam_q, cam_k)  = self.matmul2.relprop(cam1,**kwargs)
#         cam_q /= 2
#         cam_k /= 2
#
#         # #可以理解为这里获得了q k v
#         # if not self.linear:
#         #     if self.sr_ratio>1:
#         #         cam_n = rearrange()
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)
        self.softmax = Softmax(dim=-1)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        self.save_v(v)

        dots = self.matmul1([q, k]) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        out = self.matmul2([attn, v])
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def relprop(self, cam, **kwargs):
        cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DWConv(nn.Module):
    def __init__(self, dim=768,kernel=3,stride=1,padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, stride, padding, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    def relProp(self,cam,**kwargs):
        pass
class Block(nn.Module):
    def __init__(self,dim ,kernel_size=3,stride=1,padding=1, qk_scale=None,
                 mlp_ratio=4,heads=8,qkv_bias=None, drop=0.,attn_drop=0.1,drop_path=0.2,sr_ratio=1,
                 linear=False,pretrain='channels_last'):
        super(Block, self).__init__()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = Dropout(drop_path) if drop_path > 0. else nn.Identity()
        out_channels = dim*mlp_ratio or dim
        self.MLP = MLP(in_features=dim,hidden_features=out_channels, drop=drop, linear=linear)
        self.DWC = DWConvolution(dim,kernel_size,stride,padding)
        self.Attention = Attention(dim=dim,num_heads=heads,
                                         qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
        self.LN = LayerNorm(dim,data_format=pretrain)

        self.add1 = Add()
        self.add2 = Add()
        self.add3 = Add()
        self.add4 = Add()
        #bypass branch
        self.depthcon = self.DWC

        #copy
        self.clone1 = Clone()
        self.clone2 = Clone()

    def forward(self,x,H,W):

        '''
        复写这个方法 ：Clone将原始输入复制为2倍  Add就是正常的和的操作
        :param x:
        :param H:
        :param W:
        :return:
        '''
        # out = x + self.DWC(x,H,W)
        # out = self.drop_path(self.Attention(out))
        # # bypass branch
        # out = self.LN(out + self.depthcon(x,H,W))
        # out = out + x
        # out = out + self.LN(self.MLP(out,H,W))
        x1,x2,x3,x4= self.clone1(x,4)#x1是主分支上 x2主分支残差 x3侧边分支 x4是下分支与DWC配合
        x = self.add1([x1,self.DWC(x2,H,W)])
        x = self.drop_path(self.Attention(x))
        x = self.add2([self.depthcon(x3,H,W),x])#侧边分支与主分支数据之和
        x = self.LN(x)
        x = self.add3([x4,x])

        #处理MLP分支
        branchA,branchB = self.clone2(x,2)#branchA,branchB分别是主分支 有侧边分支
        x = self.add4([branchA,self.LN(self.MLP(branchB))])
        return x

    def relprop(self,cam,**kwargs):
        (_,cam2) = self.add4.relprop(cam,**kwargs)#cam1 out  cam2:MLP VALUE
        cam2 = self.MLP.relprop(self.LN.relprop(cam2,**kwargs),**kwargs)
        (cam2,_) = self.add3.relprop(cam2,**kwargs)#camout cam_x:反过来的x

        #add2
        (cam2,cam_x) = self.add2.relprop(cam2,**kwargs)
        cam_x = self.depthcon.relprop(cam_x,**kwargs)
        cam2 = self.drop_path.relprop(cam2,**kwargs)
        cam2 = self.Attention.relprop(cam2,**kwargs)
        (cam_x,cam2)=self.add1.relprop(cam_x,**kwargs)
        cam = self.DWC.relprop(cam_x,**kwargs)
        return cam
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        '''我们不在使用torch自带的ln标准化  当输入是从convolution时候使用 channels_first   当输入是一个序列时候采用 channels_last'''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))#将一个不可训练的函数转为可训练的函数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)#给定一些ln的参数
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)#实现张量和标量之间逐元素求指数操作,或者在可广播的张量之间逐元素求指数操作

            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class MultiPhraseTransformerV1(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chans=3,num_classes=1000,embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(_init_weights)
        # self.init_weights(pretrained)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def MT_b0(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),linear=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


print(MT_b0())
