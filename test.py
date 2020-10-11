# %% library
from loader import c_loader , c_loader_CIFAR10
import argparse
from model import *
import numpy as np
import torch
from tqdm import tqdm
import gc
import random



# %% test
def test(args):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    args.model.eval()
    correct = 0
    total = 0
    for s, test_batch in enumerate(tqdm(args.loader.test_iter, desc='test')):

        feature = test_batch[0].cuda(args.gpu_device)
        target = test_batch[1].cuda(args.gpu_device)
        pred = args.model(feature)
        _, predicted = torch.max(pred.data,1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    acc = (100 * correct) / total
    return acc

# %% main

def main():

    parser = argparse.ArgumentParser(description="-----[#]-----")

    # Model
    parser.add_argument('--kernel-num', type=int, default=100, help='커널 개수')
    parser.add_argument('--kernel-sizes', type=int, default=5,help='커널 사이즈')
    parser.add_argument('--class_num', type=int, default=10, help='target number')
    parser.add_argument('--input_dim', type=int, default=32, help='이미지 가로 차원 수 ')
    parser.add_argument('--input_dim_channel',type=int,default=3, help = '이미지 채널 개수')

    # Data and train
    parser.add_argument('--dataset', type=str, default='CIFAR', help='dataset CIFAR or MNIST')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training [default: 256]')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--data_size', default=50000, type=int, help='dataset size(n)')
    parser.add_argument('--experiment', type=str, default='CIFAR_temptemp', help='experiment name')

    args = parser.parse_args()

    if args.dataset == 'CIFAR':
        args.loader = c_loader_CIFAR10(args)
        print(f"{args.dataset}_loaded")
    else:
        args.loader = c_loader(args)
        print(f"{args.dataset}_loaded")

    args.model = CNN(args).cuda(args.gpu_device)
    model_parameter = args.model.state_dict()
    print('model created -- gpu version!')

    parameter = torch.load(f'./parameter/{args.experiment}.pth',
                           map_location=f'cuda:{args.gpu_device}')
    args.model.load_state_dict(parameter)
    print("loaded parameter !")

    gc.collect()
    acc = test(args)
    print(f'experiment name : {args.experiment},  accuracy : {acc}')

# %% run
if __name__ == "__main__":
    main()