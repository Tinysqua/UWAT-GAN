import torch
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_to_cuda(x, device=None):
    if device==None:
        return x.cuda()
    else:
        return x.to(device)

def ef_loss(y_true, y_pred):
    ef_loss = 0
    for i in range(len(y_true)):
        sub=torch.sub(y_true[i],y_pred[i])
        abs=torch.abs(sub)
        ef_loss += torch.mean(abs)
    return ef_loss

def ef_loss_changed(y_true, y_pred):
    l1 = nn.L1Loss()
    loss1 = l1(y_true[0], y_pred[0])
    loss2 = l1(y_true[1], y_pred[1])
    loss3 = l1(y_true[2], y_pred[2])
    loss4 = l1(y_true[3], y_pred[3])
    ef_loss = (loss1+loss2+loss3+loss4)/4
    return ef_loss

def hinge_from_tf(pred, label):
    max_value = torch.max(1-pred*label, 0)
    return torch.mean(max_value[0])


# def categorical_crossentropy(label,pred):
#     loss=0
#     for j in range(pred.shape[-1]):
#         loss += -label[j]*torch.log(pred[-1][j])
#     return loss


def pt_categorical_crossentropy(pred, label):
    return torch.sum(-label * torch.log(pred))



# Loss functions
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = convert_to_cuda(cnn)
        model = nn.Sequential()
        model = convert_to_cuda(model)
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    
    
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = convert_to_cuda(Vgg19())
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):
        x = (x+1)/2 
        y = (y+1)/2             
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
    
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()
 
    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)
    
def Discriminator_loss_computer(model_output, loss_fn, label, device_fn):
    if isinstance(model_output[0], list):
        loss = 0
        for input_i in model_output:
            pred = input_i[-1]
            target_tensor = torch.ones_like(pred) if label else torch.zeros_like(pred)
            target_tensor = device_fn(target_tensor)
            loss += loss_fn(pred, target_tensor)
    return loss

def Feat_loss_computer(model_output: tuple, num_D, n_layers, loss_fn):
    loss = 0
    pred_real = model_output[0]
    pred_fake = model_output[1]
    feat_weights = 4.0 / (n_layers+1)
    D_weights = 1.0 / num_D
    for i in range(num_D):
        for j in range(len(pred_fake[i])-1):
            loss += D_weights * feat_weights * \
                        loss_fn(pred_fake[i][j], pred_real[i][j].detach())
                        
    return loss
