import torch
from torch import nn
import math
    
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
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        super(Global_Generator, self).__init__()
        activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.n_downsampling = n_downsampling
        models = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.initialize = nn.Sequential(*models)
        bt_channel = 0
        setattr(self, 'Attention_1', CNN_Attention(ngf, norm_layer=norm_layer))
        for i in range(n_downsampling):
            mult = 2**i
            in_channel = ngf * mult
            out_channel = ngf * mult * 2
            models = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1), norm_layer(out_channel), activation]
            setattr(self, f'Down_{i}', nn.Sequential(*models))
            setattr(self, f'Attention_{i+2}', CNN_Attention(out_channel, norm_layer=norm_layer))
            bt_channel = out_channel
            
        
        models = [] 
        for i in range(n_blocks):
            models += [Branch_residual_block(bt_channel, norm_layer=norm_layer)]
        self.bt_neck = nn.Sequential(*models)
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            in_channel = ngf * mult if i == 0 else ngf * mult * 2
            out_channel = in_channel // 2 if i == 0 else in_channel // 4
            models = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(out_channel), activation]
            setattr(self, f'Up_{i}', nn.Sequential(*models))
            
        models = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.final_result = nn.Sequential(*models)
        
    def forward(self, X_input):
        X = self.initialize(X_input)
        setattr(self, 'down_result_1', X)
        for i in range(self.n_downsampling):
            Down_module = getattr(self, f'Down_{i}')
            X = Down_module(X)
            setattr(self, f'down_result_{i+2}', X)
            
        X = self.bt_neck(X)
        X_feature = 0
        for i in range(self.n_downsampling):
            Up_module = getattr(self, f'Up_{i}')
            X = Up_module(X)
            X_feature = X
            attn_module = getattr(self, f'Attention_{self.n_downsampling-i}')
            down_result = getattr(self, f'down_result_{self.n_downsampling-i}')
            X = torch.cat([X, attn_module(down_result)], dim=1)
            
        
        X = self.final_result(X)
        return X, X_feature
    
class Local_Enhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=3,
                 img_sz=256, norm_layer=nn.InstanceNorm2d, attn_sz = [64]):
        super(Local_Enhancer, self).__init__()
        activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.n_downsampling = n_downsampling
        models = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), activation]
        self.initialize = nn.Sequential(*models)
        bt_channel = 0
        setattr(self, 'Attention_1', CNN_Attention(ngf, norm_layer=norm_layer))
        for i in range(n_downsampling):
            mult = 2**i
            in_channel = ngf * mult
            out_channel = ngf * mult * 2
            models = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1), norm_layer(out_channel), activation]
            setattr(self, f'Down_{i}', nn.Sequential(*models))
            setattr(self, f'Attention_{i+2}', 
                     SelfAttention(out_channel) if img_sz in attn_sz else CNN_Attention(out_channel, norm_layer=norm_layer))
            img_sz = img_sz // 2
            bt_channel = out_channel

        models = [] 
        for i in range(n_blocks):
            models += [Branch_residual_block(bt_channel, norm_layer=norm_layer)]
        self.bt_neck = nn.Sequential(*models)
        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            in_channel = ngf * mult if i == 0 else ngf * mult * 2
            out_channel = in_channel // 2 if i == 0 else in_channel // 4
            models = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1), 
                      norm_layer(out_channel), activation]
            setattr(self, f'Up_{i}', nn.Sequential(*models))
        
        models = []
        for i in range(2):
            models += [nn.Conv2d(ngf*2, ngf*2, 3, padding='same'), norm_layer(ngf*2), activation]
        self.cache_module = nn.Sequential(*models)

        models = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.final_result = nn.Sequential(*models)
        
    def forward(self, X_input, X_feature):
        X = self.initialize(X_input)
        setattr(self, 'down_result_1', X_feature)
        for i in range(self.n_downsampling):
            Down_module = getattr(self, f'Down_{i}')
            X = Down_module(X)
            setattr(self, f'down_result_{i+2}', X)
            
        X = self.bt_neck(X)
        
        for i in range(self.n_downsampling):
            Up_module = getattr(self, f'Up_{i}')
            X = Up_module(X)
            attn_module = getattr(self, f'Attention_{self.n_downsampling-i}')
            down_result = getattr(self, f'down_result_{self.n_downsampling-i}')
            X = torch.cat([X, attn_module(down_result)], dim=1)
        
        X = self.cache_module(X)
        X = self.final_result(X)
        return X
        
        
        
        
    
if __name__ == '__main__':
    test_model = Local_Enhancer(input_nc=3, output_nc=3)
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 64, 256, 256)
    X = test_model(a, b)
    
        
        
        
        
