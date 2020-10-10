import torchvision.transforms as transforms
from torchvision.datasets import MNIST , CIFAR10
from torch.utils.data import DataLoader , random_split , Subset
import random

class loader():

    def __init__(self, args):
        super(loader, self).__init__()

        mnist_transform = transforms.Compose([transforms.ToTensor()])
        download_root = './MNIST_DATASET'

        dataset = MNIST(download_root, transform=mnist_transform, train=True, download=False)
        train_dataset , valid_dataset = random_split(dataset , [50000,10000])
        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=False)


        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)


class loader_CIFAR10():

    def __init__(self, args):
        super(loader_CIFAR10, self).__init__()

        cifar_transform = transforms.Compose([transforms.ToTensor()])
        download_root = './CIFAR10_DATASET'

        dataset = CIFAR10(download_root, transform=cifar_transform, train=True, download=False)
        train_dataset , valid_dataset = random_split(dataset , [40000,10000])
        test_dataset = CIFAR10(download_root, transform=cifar_transform, train=False, download=False)


        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)


class c_loader():

    def __init__(self, args):
        super(c_loader, self).__init__()

        mnist_transform = transforms.Compose([transforms.ToTensor()])
        download_root = './MNIST_DATASET'

        dataset = MNIST(download_root, transform=mnist_transform, train=True, download=False)
        dataset = Subset(dataset,random.sample(range(dataset.__len__()) , args.data_size))
        train_dataset , valid_dataset = random_split(dataset , [int(dataset.__len__()*0.8), dataset.__len__() - int(dataset.__len__()*0.8)])
        test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=False)

        del dataset

        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)


class c_loader_CIFAR10():

    def __init__(self, args):
        super(c_loader_CIFAR10, self).__init__()

        cifar_transform = transforms.Compose([transforms.ToTensor()])
        download_root = './CIFAR10_DATASET'

        dataset = CIFAR10(download_root, transform=cifar_transform, train=True, download=False)
        dataset = Subset(dataset,random.sample(range(dataset.__len__()) , args.data_size))
        train_dataset , valid_dataset = random_split(dataset , [int(dataset.__len__()*0.8), dataset.__len__() - int(dataset.__len__()*0.8)])
        test_dataset = CIFAR10(download_root, transform=cifar_transform, train=False, download=False)
        del dataset

        self.batch_size = args.batch_size
        self.train_iter = DataLoader(dataset=train_dataset , batch_size=self.batch_size , shuffle=True)
        self.valid_iter = DataLoader(dataset=valid_dataset , batch_size=self.batch_size , shuffle=True)
        self.test_iter = DataLoader(dataset=test_dataset , batch_size=self.batch_size , shuffle=True)


