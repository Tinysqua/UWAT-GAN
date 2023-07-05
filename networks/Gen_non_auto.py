# This is for the ablation experiment
import torch
from torch import nn
import math
from functools import partial

    
class Branch_residual_block(nn.Module):
    def __init__(self, channels, norm_layer=nn.InstanceNorm2d):
        super(Branch_residual_block, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.Pad_Conv2d_1 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, kernel_size=3, padding=0)])
        self.Pad_Conv2d_2 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, kernel_size=3, padding=0)])
        self.Pad_Conv2d_3 = nn.Sequential(*[nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, kernel_size=3, padding=0)])
        
        self.norm_layer_1 = norm_layer(channels)
        self.norm_layer_2 = norm_layer(channels)
        self.norm_layer_3 = norm_layer(channels)
        
    def forward(self, X_input):
        X = X_input
        X = self.Pad_Conv2d_1(X)
        X = self.norm_layer_1(X)
        X = self.LeakyReLU(X)
        
        X_branch_1 = self.Pad_Conv2d_2(X)
        X_branch_1 = self.norm_layer_2(X_branch_1)
        X_branch_1 = self.LeakyReLU(X_branch_1)
        
        X_branch_2 = self.Pad_Conv2d_3(X)
        X_branch_2 = self.norm_layer_3(X_branch_2)
        X_branch_2 = self.LeakyReLU(X_branch_2)
        X_add_branch_1_2 = torch.add(X_branch_2, X_branch_1)
        X_final = torch.add(X_input, X_add_branch_1_2)
        return X_final
    
class CNN_Attention(nn.Module):
    def __init__(self, channels, norm_layer=nn.InstanceNorm2d):
        super(CNN_Attention, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.norm_layer_1 = norm_layer(channels)
        self.norm_layer_2 = norm_layer(channels)
        
        self.Conv_3_1_first = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=1)
        self.Conv_3_1_second = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=1)
        
    def forward(self, X):
        X_input = X
        X = self.Conv_3_1_first(X)
        X = self.norm_layer_1(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        
        X = self.Conv_3_1_second(X)
        X = self.norm_layer_2(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        return X
    
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

 
class Global_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d, attn_con=True):
        super(Global_Generator, self).__init__()
        activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.tensor_cat = partial(torch.cat, dim=1)
        models = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.initialize = nn.Sequential(*models)
        self.Attention_1 = CNN_Attention(ngf, norm_layer=norm_layer) if attn_con else nn.Identity()
        
        models = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]
        self.Down_0 = nn.Sequential(*models)
        self.Attention_2 = CNN_Attention(ngf*2, norm_layer=norm_layer) if attn_con else nn.Identity()
        
        models = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]
        self.Down_1 = nn.Sequential(*models)
        self.Attention_3 = CNN_Attention(ngf*4, norm_layer=norm_layer) if attn_con else nn.Identity()
        
        models = [nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]
        self.Down_2 = nn.Sequential(*models)
        
        models = [] 
        for i in range(n_blocks):
            models += [Branch_residual_block(ngf*8, norm_layer=norm_layer)]
        self.bt_neck = nn.Sequential(*models)
        
        models = [nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(ngf*4), activation]
        self.Up_0 = nn.Sequential(*models)
        
        models = [nn.ConvTranspose2d(ngf*8, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(ngf*4), activation]
        self.Up_1 = nn.Sequential(*models)
        
        models = [nn.ConvTranspose2d(ngf*4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(ngf*4), activation]
        self.Up_2 = nn.Sequential(*models)
        
        models = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.final_result = nn.Sequential(*models)
        
    def forward(self, X_input):
        X = self.initialize(X_input)
        down_result_1 = X
        X = self.Down_0(X)
        down_result_2 = X
        X = self.Down_1(X)
        down_result_3 = X
        X = self.Down_2(X)
        
        X = self.bt_neck(X)
        
        X = self.Up_0(X)
        X = self.tensor_cat([X, self.Attention_3(down_result_3)])
        X = self.Up_1(X)
        X = self.tensor_cat([X, self.Attention_2(down_result_2)])
        X = self.Up_2(X)
        X = self.tensor_cat([X, self.Attention_1(down_result_1)])
        
        X_feature = X
        X = self.final_result(X)
        return X, X_feature
    
class Local_Enhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=3,
                 norm_layer=nn.InstanceNorm2d, attn_con = True):
        super(Local_Enhancer, self).__init__()
        activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.tensor_cat = partial(torch.cat, dim=1)
        models = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.initialize = nn.Sequential(*models)
        self.Attention_1 = CNN_Attention(ngf, norm_layer=norm_layer) if attn_con else nn.Identity()
        
        models = [nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1), norm_layer(ngf*2), activation]
        self.Down_0 = nn.Sequential(*models)
        self.Attention_2 = CNN_Attention(ngf*2, norm_layer=norm_layer) if attn_con else nn.Identity()
        
        models = [nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1), norm_layer(ngf*4), activation]
        self.Down_1 = nn.Sequential(*models)
        
        
        models = [] 
        for i in range(n_blocks):
            models += [Branch_residual_block(ngf*4, norm_layer=norm_layer)]
        self.bt_neck = nn.Sequential(*models)
        
        models = [nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(ngf*2), activation]
        self.Up_0 = nn.Sequential(*models)
        
        models = [nn.ConvTranspose2d(ngf*4, ngf, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(ngf), activation]
        self.Up_1 = nn.Sequential(*models)
        
        
        models = []
        models += [nn.Conv2d(ngf*4, ngf*2, 3, padding='same'), norm_layer(ngf*2), activation]
        models += [nn.Conv2d(ngf*2, ngf*2, 3, padding='same'), norm_layer(ngf*2), activation]
        # for i in range(2):
        #     models += [nn.Conv2d(ngf*2, ngf*2, 3, padding='same'), norm_layer(ngf*2), activation]
        self.cache_module = nn.Sequential(*models)
        
        models = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.final_result = nn.Sequential(*models)
        
    def forward(self, X_input, X_feature):
        X = self.initialize(X_input)
        self.down_result_1 = X
        X = self.Down_0(X)
        self.down_result_2 = X
        X = self.Down_1(X)
        
        X = self.bt_neck(X)
        
        X = self.Up_0(X)
        X = self.tensor_cat([X, self.Attention_2(self.down_result_2)])
        X = self.Up_1(X)
        X = self.tensor_cat([X, self.Attention_1(self.down_result_1), X_feature])
        
        X = self.cache_module(X)
        X = self.final_result(X)
        
        return X
        
        
        
        