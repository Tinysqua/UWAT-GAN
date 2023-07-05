import torch.nn.functional as F
import torch
from einops import rearrange
from torch import nn
import numpy as np

class PatchEncoder(torch.nn.Module):
    def __init__(self, num_patches=64, projection_dim=64, patch_size=64):
        super(PatchEncoder, self).__init__()
        self.num_patches= num_patches
        self.projection = torch.nn.Linear(patch_size*patch_size*4,out_features=projection_dim)  #256*256:262144
        # self.projection_resize = torch.nn.Linear(32*32*4, out_features=projection_dim)
        self.position_embedding = torch.nn.Embedding(num_patches,projection_dim)

    def forward(self, input):
        
        # positions = torch.nn.Parameter(torch.randn(1, self.num_patches, self.num_patches)).cuda()
        positions = torch.arange(self.num_patches).cuda()
        # if input.shape[-1]==4096:
        #     encoded = self.projection_resize(input)+self.position_embedding(positions)
        # else:
        encoded = self.projection(input) + self.position_embedding(positions)
        return encoded
        # if input.shape[-1]==4096:
        #     encoded = self.projection_resize(input)+self.position_embedding(positions)
        # else:
        #      encoded = self.projection(input) + self.position_embedding(positions)
        # return encoded
        
class Block(torch.nn.Module):
    def __init__(self, project_dim, depth, num_heads, mlp_ratio):
        super(Block, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.normlayer1 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.normlayer2 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.linear1 = torch.nn.Linear(project_dim, project_dim * mlp_ratio)
        self.linear2 = torch.nn.Linear(project_dim * mlp_ratio, project_dim)
        self.gelu = torch.nn.GELU()
        for i in range(depth):
            setattr(self, "layer"+str(i+1), torch.nn.MultiheadAttention(project_dim, num_heads, dropout=0.1))
    
    def forward(self, encoded_patches):
        feat = []
        for i in range(self.depth):
            x1 = self.normlayer1(encoded_patches)
            attention_output, attn_output_weights = getattr(self, "layer"+str(i+1))(x1, x1, x1)
            x2 = encoded_patches + attention_output
            x3 = self.normlayer2(x2)
            x3 = self.mlp(x3)
            encoded_patches = x2 + x3
            feat.append(encoded_patches)
        feat_total = [feat[0], feat[1], feat[2], feat[3]]
        return feat_total, encoded_patches
            
            
            
    def mlp(self, x, dropout_rate=0.1):
        x = self.linear1(x)
        x = self.gelu(x)
        x = F.dropout(x, p=dropout_rate)
        x = self.linear2(x)
        x = self.gelu(x)
        x = F.dropout(x, p=dropout_rate)
        return x
    

# patchsize: 64 
class vit_discriminator(torch.nn.Module):
    def __init__(self, patch_size, project_dim=64,num_heads=4, mlp_ratio=2, depth=4, img_size=512):
        super(vit_discriminator, self).__init__()
        self.patch_size = patch_size
        self.GELU = torch.nn.GELU()
        self.block = Block(project_dim=project_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio).cuda()
        self.Conv_4_1 = torch.nn.Conv2d(1, 1, (4, 4), padding='same')
        self.Conv_4_1_2 = torch.nn.Conv2d(1, 64, (4,4), padding='same')
        # self.Conv_4_1 = torch.nn.Conv2d(1, 1, (3, 3), padding=1)
        # self.Conv_4_1_2 = torch.nn.Conv2d(1, 64, (3,3), padding=1)
        self.MultiHeadAttention = torch.nn.MultiheadAttention(project_dim, num_heads, dropout=0.1)
        #self.LayerNorm = torch.nn.LayerNorm(64, eps=1e-6)
        self.linear3 = torch.nn.Linear(64, 2)
        self.Softmax = torch.nn.Softmax(dim=-1)
        self.AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        if isinstance(img_size, int):
            self.num_patches = (img_size // patch_size) ** 2
        else:
            self.num_patches = img_size[0]//patch_size * img_size[1]//patch_size
            
        self.PatchEncoder=PatchEncoder(num_patches=self.num_patches, projection_dim=project_dim, patch_size=self.patch_size).cuda()
        self.LayerNormalization_0 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.LayerNormalization_1 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)
        self.LayerNormalization_2 = torch.nn.LayerNorm(normalized_shape=project_dim, eps=1e-6)


    def forward(self,fundus,angio):
        patch_size = self.patch_size
        X = torch.cat((fundus, angio), 1)
        feat = []
        patches = rearrange(X, 'b c (h h1) (w w2) -> b (h w) (h1 w2 c)', h1=patch_size, w2=patch_size)
        encoded_patches = self.PatchEncoder(patches)
        
        feat, encoded_patches = self.block(encoded_patches)

        representation = self.LayerNormalization_0(encoded_patches)
        
        X_reshape = representation.unsqueeze(0).permute(1,0,2,3)
        X = self.Conv_4_1(X_reshape)
        out_hinge = torch.tanh(X)
        representation = self.Conv_4_1_2(X_reshape)
        features = self.AdaptiveAvgPool2d(representation).squeeze(-1).squeeze(-1)
        classses = self.linear3(features)
        out_class = self.Softmax(classses)
        return [out_hinge, out_class, feat]
    

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


