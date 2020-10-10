# %% library
from loader import c_loader , c_loader_CIFAR10
import argparse
from model import *
import numpy as np
import torch
import torch.optim as optim
from pytorchtools import EarlyStopping
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
import random
import matplotlib.pyplot as plt



# %% Train
def train(args, train_loss_list, valid_loss_list):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    optimizer = optim.Adam(args.model.parameters(), args.learning_rate)

    # criterion = nn.MSELoss(reduction='mean')
    criterion = my_loss

    best_MSE = np.inf
    start = time.time()

    writer = SummaryWriter(f'./runs/{args.experiment}')
    early_stopping = EarlyStopping(patience= 15, verbose=False, path = f'./parameter/{args.experiment}.pth')

    for e in range(args.epoch):
        print("\n===> epoch %d" % e)

        total_loss = 0

        for i, batch in enumerate(tqdm(args.loader.train_iter, desc='train')):


            feature = batch[0].cuda(args.gpu_device)
            # target = batch[1].cuda(args.gpu_device)
            optimizer.zero_grad()
            args.model.train()
            pred = args.model(feature)
            loss = criterion(pred, feature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            if (i + 1) % args.printevery == 0:

                with torch.no_grad():

                    args.model.eval()
                    val_loss = 0

                    for s, val_batch in enumerate(tqdm(args.loader.valid_iter, desc='valid')):

                        feature = batch[0].cuda(args.gpu_device)
                        # target = batch[1].cuda(args.gpu_device)
                        pred = args.model(feature)

                        v_loss = criterion(pred, feature)
                        val_loss += v_loss.data.item()

                if best_MSE > (val_loss / len(args.loader.valid_iter)):
                    best_MSE = (val_loss / len(args.loader.valid_iter))
                    torch.save(args.model.state_dict(), f'./parameter/best_parameter_{args.experiment}.pth')

                iters = (e) * (len(args.loader.train_iter)) + i
                avg_loss = total_loss / args.printevery
                train_loss_list.append(avg_loss)
                valid_loss_list.append(val_loss / len(args.loader.valid_iter))

                writer.add_scalar('train_loss', avg_loss, iters+1)
                writer.add_scalar('valid_loss', val_loss / len(args.loader.valid_iter), iters+1)
                # total_loss = 0
                # if args.dataset == "MNIST":
                #     show_visual_progress(args, rows=5, title=f'{args.experiment}_{iters}')
                #     show_filter_progress(args, title=f'{args.experiment}_{iters}')
                # else:
                #     show_visual_progress(args, rows=3, title=f'{args.experiment}_{iters}')
                #     show_filter_progress(args, title=f'{args.experiment}_{iters}')
                # plt.close('all')
                early_stopping(val_loss / len(args.loader.valid_iter) , args.model)

                if early_stopping.early_stop:
                    print('Early stopping')
                    break

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


def show_visual_progress(args, rows=5, title=None):

    if args.dataset == "MNIST":

        fig = plt.figure(figsize=(10, 8))
        if title:
            plt.title(title)

        image_rows = []
        for idx, (feature, label) in enumerate(args.loader.test_iter):
            if rows == idx:
                break
            feature = feature.cuda(args.gpu_device)
            images = args.model(feature).detach().cpu().numpy().reshape(feature.size(0),args.input_dim, args.input_dim)
            images_idxs = [list(label.numpy()).index(x) for x in range(10)]
            combined_images = np.concatenate([images[x].reshape(args.input_dim, args.input_dim) for x in images_idxs], 1)
            image_rows.append(combined_images)

        plt.imshow(np.concatenate(image_rows))
        plt.savefig('./img/'+title+'.png',dpi=300)
        # plt.show()

    else:

        fig = plt.figure(figsize=(20,16))
        if title:
            plt.title(title)

        image_rows = []
        for idx, (feature, label) in enumerate(args.loader.test_iter):
            if rows == idx:
                break
            feature = feature.cuda(args.gpu_device)
            images = args.model(feature).detach().cpu().numpy().reshape(feature.size(0),args.input_dim_channel,args.input_dim, args.input_dim)
            images_idxs = [list(label.numpy()).index(x) for x in np.random.randint(0,10,5)]
            combined_images = np.concatenate([images[x].reshape(args.input_dim_channel,args.input_dim, args.input_dim) for x in images_idxs], 1)
            image_rows.append(combined_images)
        
        img = np.concatenate(image_rows, axis=2)
        img = img / np.amax(img)
        img = np.clip(img,0,1)
        plt.imshow(img.reshape(rows*args.input_dim, args.input_dim*5, args.input_dim_channel))
        plt.savefig('./img/'+title+'.png',dpi=500)
        # plt.show()






def show_filter_progress(args,title=None):

    if args.dataset == 'CIFAR':

        conv1 = args.model.state_dict()['conv1.weight'].detach().cpu().numpy().reshape(3,100,5,5)

        for chanel_ind , chanel in enumerate(conv1):

            fig = plt.figure(figsize=(10,3))
            for ind, img in enumerate(chanel):
                ax = fig.add_subplot(5, 20, ind + 1)
                ax.imshow(img, interpolation='nearest',cmap = 'gray')
                ax.set_xticks([]), ax.set_yticks([])
                # ax.set_title(f'{chanel_ind}_filter_{ind + 1}')
            # fig.set_size_inches(np.array(fig.get_size_inches()) * 10)
            # plt.tight_layout()
            plt.savefig(f'./filter/'+title+f'_{chanel_ind}.png',dpi=300)
    else:

        conv1 = args.model.state_dict()['conv1.weight'].detach().cpu().numpy().reshape(20, 7, 7)
        fig = plt.figure(figsize=(10, 3))

        for ind, img in enumerate(conv1):
            ax = fig.add_subplot(2, 10, ind + 1)
            ax.imshow(img, interpolation='nearest',cmap = 'gray')
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f'filter_{ind + 1}')
        # fig.set_size_inches(np.array(fig.get_size_inches()) * 10)
        plt.subplots_adjust(bottom=0.1,top=0.1,left=0.1,right=0.1,wspace=0.1,hspace=0.1)
        plt.tight_layout()
        plt.savefig('./filter/'+title+'.png',dpi=300)


# %% main
def main():
    parser = argparse.ArgumentParser(description="-----[#]-----")

    # Model
    parser.add_argument('--kernel-num', type=int, default=100, help='커널 개수')
    parser.add_argument('--kernel-sizes', type=int, default=5,help='커널 사이즈')
    parser.add_argument('--class_num', type=int, default=10, help='target number')
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epoch", default=30, type=int, help="number of max epoch")
    parser.add_argument('--input_dim', type=int, default=28, help='이미지 가로 차원 수 ')
    parser.add_argument('--input_dim_channel',type=int,default=1, help = '이미지 채널 개수')

    # Data and train
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset CIFAR or MNIST')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--printevery', default=100, type=int, help='log , print every % iteration')
    parser.add_argument('--data_size', default=50000, type=int, help='dataset size(n)')
    parser.add_argument('--experiment', type=str, default='MNIST_ready_fine_tune_datasize_50000', help='experiment name')

    args = parser.parse_args()

    if args.dataset == 'CIFAR':
        args.loader = c_loader_CIFAR10(args)
        print(f"{args.dataset}_loaded")
    else:
        args.loader = c_loader(args)
        print(f"{args.dataset}_loaded")

    args.model = CNN_for_cae(args).cuda(args.gpu_device)
    print(f'model{args.experiment} created -- gpu version!')

    train_loss_list = []
    valid_loss_list = []
    gc.collect()
    train(args, train_loss_list, valid_loss_list)


# %% run
if __name__ == "__main__":
    main()