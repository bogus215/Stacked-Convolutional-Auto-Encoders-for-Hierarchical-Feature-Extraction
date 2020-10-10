#%% lib
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from skimage.util import random_noise


#%% CAE 1
class CAE1(nn.Module):
    def __init__(self,args):
        super(CAE1, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.Ks = args.kernel_sizes
        self.b_e = nn.Parameter(torch.randn(self.Co))
        self.b_d = nn.Parameter(torch.randn(self.img_dim))
        self.conv1 = nn.Conv2d(self.img_dim, self.Co , (self.Ks,self.Ks) , bias=False)
        self.conv2 = nn.ConvTranspose2d(self.Co,self.img_dim,(self.Ks,self.Ks), bias=False)


    def forward(self,x):
        x = self.conv1(x) + self.b_e.view(1, self.Co,1,1)
        x = x * x.tanh()
        x = self.conv2(x) + self.b_d.view(1,self.img_dim,1,1)
        x = x * x.tanh()

        return x

#%% CAE 2
class CAE2(nn.Module):
    def __init__(self,args):
        super(CAE2, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.Ks = args.kernel_sizes
        self.b_e = nn.Parameter(torch.randn(self.Co))
        self.b_d = nn.Parameter(torch.randn(self.img_dim))

        self.conv1 = nn.Conv2d(self.img_dim, self.Co , (self.Ks,self.Ks) , bias=False)
        self.conv2 = nn.ConvTranspose2d(self.Co,self.img_dim,(self.Ks,self.Ks), bias=False)

        if args.dataset == 'MNIST':
            self.noise = binary_noise(p=0.5)
        else:
            self.noise = gaussian_noise(args ,var=0.05, is_relative_detach=True)


    def forward(self,x):
        x = self.noise(x)
        x = self.conv1(x) + self.b_e.view(1, self.Co, 1, 1)
        x = x * x.tanh()
        x = self.conv2(x) + self.b_d.view(1,self.img_dim,1,1)
        x = x * x.tanh()

        return x

#%% CAE 3
class CAE3(nn.Module):
    def __init__(self,args):
        super(CAE3, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.Ks = args.kernel_sizes
        self.b_e = nn.Parameter(torch.randn(self.Co))
        self.b_d = nn.Parameter(torch.randn(self.img_dim))

        self.conv1 = nn.Conv2d(self.img_dim, self.Co , (self.Ks,self.Ks) , bias=False)
        self.max = nn.MaxPool2d((2,2) , stride=2 , return_indices=True)
        self.unmax = nn.MaxUnpool2d((2,2) , stride=2)
        self.conv2 = nn.ConvTranspose2d(self.Co,self.img_dim,(self.Ks,self.Ks), bias=False)


    def forward(self,x):
        x = self.conv1(x) + self.b_e.view(1, self.Co, 1, 1)
        x = x * x.tanh()
        x , indices = self.max(x)
        x = self.unmax(x , indices)
        x = self.conv2(x) + self.b_d.view(1,self.img_dim,1,1)
        x = x * x.tanh()

        return x


# %% CAE 4
class CAE4(nn.Module):
    def __init__(self, args):
        super(CAE4, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.Ks = args.kernel_sizes
        self.b_e = nn.Parameter(torch.randn(self.Co))
        self.b_d = nn.Parameter(torch.randn(self.img_dim))

        if args.dataset == 'MNIST':
            self.noise = binary_noise(p=0.7)
        else:
            self.noise = gaussian_noise(args ,var=0.05, is_relative_detach=True)

        self.conv1 = nn.Conv2d(self.img_dim, self.Co , (self.Ks,self.Ks) , bias=False)
        self.max = nn.MaxPool2d((2,2) , stride=2 , return_indices=True)
        self.unmax = nn.MaxUnpool2d((2,2) , stride=2)
        self.conv2 = nn.ConvTranspose2d(self.Co,self.img_dim,(self.Ks,self.Ks), bias=False)


    def forward(self, x):
        x = self.noise(x)
        x = self.conv1(x) + self.b_e.view(1, self.Co, 1, 1)
        x = x * x.tanh()
        x , indices = self.max(x)
        x = self.unmax(x , indices)
        x = self.conv2(x) + self.b_d.view(1,self.img_dim,1,1)
        x = x * x.tanh()

        return x

# %% CNN
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.conv1 = nn.Conv2d(self.img_dim,100,(5,5))
        self.max1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(100,150,(5,5))
        self.max2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(150,200,(3,3))

        if args.dataset == "MNIST":
            self.classifier = nn.Sequential(nn.Linear(800, 300),nn.ReLU(),
                                            nn.Linear(300, args.class_num))

        else:
            self.classifier = nn.Sequential( nn.Linear(1800,300),
                                             nn.ReLU(),
                                             nn.Linear(300,args.class_num))

    def forward(self,x):
        x = self.conv1(x)
        x = x * x.tanh()
        x = self.max1(x)
        x = self.conv2(x)
        x = x * x.tanh()
        x = self.max2(x)
        x = self.conv3(x)
        x = x * x.tanh()
        x = x.view(-1,np.prod(x.size()[1:]))
        x = self.classifier(x)

        return F.softmax(x,dim =1)


# %% CNN for CAE
class CNN_for_cae(nn.Module):
    def __init__(self, args):
        super(CNN_for_cae, self).__init__()

        self.img = args.input_dim
        self.img_dim = args.input_dim_channel
        self.Co = args.kernel_num
        self.Ks = args.kernel_sizes
        self.b_e1 = nn.Parameter(torch.randn(100))
        self.b_e2 = nn.Parameter(torch.randn(150))
        self.b_e3 = nn.Parameter(torch.randn(200))

        self.b_d1 = nn.Parameter(torch.randn(150))
        self.b_d2 = nn.Parameter(torch.randn(100))
        self.b_d3 = nn.Parameter(torch.randn(self.img_dim))

        if args.dataset == 'MNIST':
            self.noise = binary_noise(p=0.7)
        else:
            self.noise = gaussian_noise(args ,var=0.05, is_relative_detach=True)

        self.conv1 = nn.Conv2d(self.img_dim,100, (5,5) , bias=False)
        self.conv2 = nn.Conv2d(100,150, (5,5) , bias=False)
        self.conv3 = nn.Conv2d(150,200, (3,3) , bias=False)
        self.max = nn.MaxPool2d(2,stride=2 , return_indices=True)
        self.unmax = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(200,150,(3,3), bias=False)
        self.deconv2 = nn.ConvTranspose2d(150,100,(5,5), bias=False)
        self.deconv3 = nn.ConvTranspose2d(100,self.img_dim,(5,5), bias=False)

    def forward(self,x):
        x = self.noise(x)
        x = self.conv1(x) + self.b_e1.view(1, 100, 1, 1)
        x = x * x.tanh()
        x , indices = self.max(x)
        x = self.unmax(x , indices)

        x = self.conv2(x) + self.b_e2.view(1, 150, 1, 1)
        x = x * x.tanh()
        x , indices = self.max(x)
        x = self.unmax(x , indices)

        x = self.conv3(x) + self.b_e3.view(1, 200, 1, 1)
        x = x * x.tanh()
        x , indices = self.max(x)
        x = self.unmax(x , indices)

        x = self.deconv1(x) + self.b_d1.view(1,150,1,1)
        x = x * x.tanh()
        x = self.deconv2(x) + self.b_d2.view(1,100,1,1)
        x = x * x.tanh()
        x = self.deconv3(x) + self.b_d3.view(1,self.img_dim,1,1)
        x = x * x.tanh()

        return x


def my_loss(output,target):
    loss = torch.mean((output - target) ** 2)
    return loss


class binary_noise(nn.Module):
    def __init__(self,p):
        super(binary_noise, self).__init__()

        self.p = p

    def forward(self,img):
        random = torch.FloatTensor(img.shape).uniform_(0, 1) > self.p
        img[random] = 0
        return img


class gaussian_noise(nn.Module):
    def __init__(self,args,var,is_relative_detach=True):
        super(gaussian_noise, self).__init__()

        self.var = var
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda(args.gpu_device)

    def forward(self,img):
        scale = self.var * img.detach() if self.is_relative_detach else self.sigma * img
        sampled_noise = self.noise.repeat(*img.size()).float().normal_()*scale
        img = img + sampled_noise
        return img