# %% library
from loader import c_loader , c_loader_CIFAR10
import argparse
from model import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from tqdm import tqdm
import gc
import random



# %% Train
def train(args, train_loss_list, valid_loss_list):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


    optimizer = optim.Adam(args.model.parameters(), args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    best_MSE = np.inf
    start = time.time()

    writer = SummaryWriter(f'./runs/{args.experiment}')
    early_stopping = EarlyStopping(patience= 15, verbose=False, path = f'./parameter/{args.experiment}.pth')

    for e in range(args.epoch):
        print("\n===> epoch %d" % e)

        total_loss = 0

        for i, batch in enumerate(tqdm(args.loader.train_iter, desc='train')):


            feature = batch[0].cuda(args.gpu_device)
            target = batch[1].cuda(args.gpu_device)
            optimizer.zero_grad()
            args.model.train()
            pred = args.model(feature)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            if (i + 1) % args.printevery == 0:

                with torch.no_grad():

                    args.model.eval()
                    val_loss = 0
                    correct = 0
                    total = 0
                    for s, val_batch in enumerate(tqdm(args.loader.valid_iter, desc='valid')):

                        feature = batch[0].cuda(args.gpu_device)
                        target = batch[1].cuda(args.gpu_device)
                        pred = args.model(feature)
                        v_loss = criterion(pred, target)
                        val_loss += v_loss.data.item()
                        _, predicted = torch.max(pred.data,1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

                if best_MSE > (val_loss / len(args.loader.valid_iter)):
                    best_MSE = (val_loss / len(args.loader.valid_iter))
                    torch.save(args.model.state_dict(), f'./parameter/best_parameter_{args.experiment}.pth')

                iters = (e) * (len(args.loader.train_iter)) + i
                avg_loss = total_loss / args.printevery
                train_loss_list.append(avg_loss)
                valid_loss_list.append(val_loss / len(args.loader.valid_iter))
                val_acc = 100 * correct /total
                writer.add_scalar('train_loss', avg_loss, iters+1)
                writer.add_scalar('valid_loss', val_loss / len(args.loader.valid_iter), iters+1)
                writer.add_scalar('valid_acc', val_acc, iters + 1)
                early_stopping(-val_acc , args.model)
                if early_stopping.early_stop:
                    print('Early stopping')
                    break
                print('epoch:', e + 1, '/ train_loss:', avg_loss, '/ valid_loss:',
                      val_loss / len(args.loader.valid_iter), '/', (time.time() - start) // 60, 'm')
                total_loss = 0

        if early_stopping.early_stop:
            print('Early stopping')
            break

        with open(f'./loss/train_loss_{args.experiment}.pickle', 'wb') as f:
            pickle.dump(train_loss_list, f)

        with open(f'./loss/valid_loss_{args.experiment}.pickle', 'wb') as f:
            pickle.dump(valid_loss_list, f)

        # torch.save(args.model.state_dict(), f'./parameter/{e}_parameter_{args.experiment}.pth')

        for name, param in args.model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), e)


# %% main

def main():

    parser = argparse.ArgumentParser(description="-----[#]-----")

    # Model
    parser.add_argument('--kernel-num', type=int, default=100, help='커널 개수')
    parser.add_argument('--kernel-sizes', type=int, default=5,help='커널 사이즈')
    parser.add_argument('--class_num', type=int, default=10, help='target number')
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epoch", default=300, type=int, help="number of max epoch")
    parser.add_argument('--input_dim', type=int, default=28, help='이미지 가로 차원 수 ')
    parser.add_argument('--input_dim_channel',type=int,default=1, help = '이미지 채널 개수')

    # Data and train
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset CIFAR or MNIST')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--printevery', default=100, type=int, help='log , print every % iteration')
    parser.add_argument('--data_size', default=50000, type=int, help='dataset size(n)')
    parser.add_argument('--experiment', type=str, default='MNIST', help='experiment name')

    args = parser.parse_args()

    if args.dataset == 'CIFAR':
        args.loader = c_loader_CIFAR10(args)
        print(f"{args.dataset}_loaded")
    else:
        args.loader = c_loader(args)
        print(f"{args.dataset}_loaded")

    args.model = CNN(args).cuda(args.gpu_device)
    print('model created -- gpu version!')
    train_loss_list = []
    valid_loss_list = []
    gc.collect()
    train(args, train_loss_list, valid_loss_list)


# %% run
if __name__ == "__main__":
    main()