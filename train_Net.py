"""
Convolutional Encoder-Decoder Networks for Image-to-Image Regression

"""

from dense_ed import DenseEDS, DenseED
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
import h5py
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from time import time

import pandas as pd
from scipy import stats, integrate
import seaborn as sns
plt.switch_backend('agg')

# default to use cuda
parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='Net', help='experiment name')
parser.add_argument('--skip', action='store_true', default=False, help='enable skip connection between encoder and decoder nets')
parser.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

parser.add_argument('--data-dir', type=str, default="/afs/crc.nd.edu/user/s/smo/invers_mt3d/", help='data directory')
parser.add_argument('--kle-terms', type=int, default=679, help='num of KLE terms')
parser.add_argument('--n-train', type=int, default=400, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")

parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005, help='learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=30, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=50, help='input batch size for testing (default: 100)')
parser.add_argument('--log-interval', type=int, default=5, help='how many epochs to wait before logging training status')
parser.add_argument('--plot-interval', type=int, default=50, help='how many epochs to wait before plotting training status')

args = parser.parse_args()
device = th.device("cuda" if th.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


all_over_again = 'experiments/April_16_test'

exp_dir = args.data_dir + all_over_again + "/{}/kle_{}/Ntrs{}__Bks{}_Bts{}_Eps{}_wd{}_lr{}_K{}".\
    format(args.exp_name, args.kle_terms, args.n_train,args.blocks,
           args.batch_size, args.n_epochs, args.weight_decay, args.lr, args.growth_rate)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# load training data
hdf5_dir = args.data_dir + "raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_train)
with h5py.File(hdf5_dir + "/input_lhs{}.hdf5".format(args.n_train), 'r') as f:
    x_train = f['dataset'][()]
    print("total input data shape: {}".format(x_train.shape))
with h5py.File(hdf5_dir + "/output_lhs{}.hdf5".format(args.n_train), 'r') as f:
    y_train = f['dataset'][()]
    print("total output data shape: {}".format(y_train.shape))
# load test data
hdf5_dir = args.data_dir + "raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_test)
with h5py.File(hdf5_dir + "/input_lhs{}.hdf5".format(args.n_test), 'r') as f:
    x_test = f['dataset'][()]
    print("test input data shape: {}".format(x_test.shape))
with h5py.File(hdf5_dir + "/output_lhs{}.hdf5".format(args.n_test), 'r') as f:
    y_test = f['dataset'][()]
    print("output data shape: {}".format(y_test.shape))

y_train_mean = np.mean(y_train, 0)
y_train_var = np.sum((y_train - y_train_mean) ** 2)
print('y_train_var: {}'.format(y_train_var))
train_stats = {}
train_stats['y_mean'] = y_train_mean
train_stats['y_var'] = y_train_var

y_test_mean = np.mean(y_test, 0)
y_test_var = np.sum((y_test - y_test_mean) ** 2)
print('y_test_var: {}'.format(y_test_var))
test_stats = {}
test_stats['y_mean'] = y_test_mean
test_stats['y_var'] = y_test_var

kwargs = {'num_workers': 4,
          'pin_memory': True} if th.cuda.is_available() else {}

data_train = th.utils.data.TensorDataset(th.FloatTensor(x_train),
                                                th.FloatTensor(y_train))
data_test = th.utils.data.TensorDataset(th.FloatTensor(x_test),
                                                th.FloatTensor(y_test))
train_loader = th.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
test_loader = th.utils.data.DataLoader(data_test,
                                               batch_size=args.test_batch_size,
                                               shuffle=True, **kwargs)

model = DenseED(x_train.shape[1], y_train.shape[1], blocks=args.blocks, growth_rate=args.growth_rate,
                drop_rate=args.drop_rate, bn_size=args.bn_size,
                num_init_features=args.init_features, bottleneck=args.bottleneck).to(device)

print(model)
print("number of parameters: {}\nnumber of layers: {}"
              .format(*model._num_parameters_convlayers()))

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)

n_out_pixels_train = len(train_loader.dataset) * train_loader.dataset[0][1].numel()
n_out_pixels_test = len(test_loader.dataset) * test_loader.dataset[0][1].numel()


def test(epoch, plot_intv=25):
    model.eval()
    loss = 0.
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
        with th.no_grad():
            output = model(input)
        loss += F.mse_loss(output, target,size_average=False).item()

        # plot predictions
        if epoch % plot_intv == 0 and batch_idx == len(test_loader) - 1:
            n_samples = 4
            idx = th.LongTensor(np.random.choice(args.n_test, n_samples, replace=False))

            for i in range(n_samples):
                x = x_test[ [idx[i]] ]
                samples_target = y_test[ idx[i] ]
                x_tensor = (th.FloatTensor(x)).to(device)
                y_hat = model(x_tensor)
                samples_output = y_hat[0].data.cpu().numpy()
                samples_err = samples_target - samples_output
                samples = np.vstack((samples_target, samples_output, samples_err))

                Nout = y_test.shape
                c_max = np.full( (Nout*3), 0.0)
                for l in range(Nout*3):
                    if l < Nout:
                        c_max[l] = np.max(samples[l])
                    elif Nout <= l < 2*Nout:
                        c_max[l] = np.max(samples[l])
                        if c_max[l] > c_max[l-Nout]:
                            c_max[l-Nout] = c_max[l]
                        else:
                            c_max[l] = c_max[l-Nout]
                    else:
                        c_max[l] = np.max( np.abs(samples[l]) )
                c_max = np.loadtxt("Cmax.dat")

                LetterId = (['a','b','c','d', 'e','f','g','h', 'i','j','k','m'])
                ylabel = (['$\mathbf{y}$', '$\hat{\mathbf{y}}$', '$\mathbf{y}-\hat{\mathbf{y}}$'])
                fig = plt.figure(figsize=(4*4-0.5, 10))
                outer = gridspec.GridSpec(2, 1, wspace=0.01, hspace=0.06)
                nl = 40
                m = 0
                samp_id = [ [0,1,2,3, 8,9,10,11, 16,17,18,19], [4,5,6,7, 12,13,14,15, 20,21,22,23] ]
                for j in range(2):
                    inner = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec = outer[j], wspace=0.2, hspace=0.08)
                    l = 0
                    for k in range(3*4):
                        ax = plt.Subplot(fig, inner[k])
                        ax.set_aspect('equal')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        s_id = samp_id[j][k]
                        if k < 2*4:
                            cax = ax.contourf(samples[ s_id], np.arange(0.0 , c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet')
                            fig.add_subplot(ax)
                        else:
                            cax = ax.contourf(samples[ s_id], np.arange(0.0 - c_max[s_id] - c_max[s_id]/nl*1, c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet',extend='both')
                            fig.add_subplot(ax)
                        ax.spines['left'].set_color('white')
                        ax.spines['right'].set_color('white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['top'].set_color('white')
                        cbar = plt.colorbar(cax, ax=ax, fraction=0.021, pad=0.04,
                                            format=ticker.FuncFormatter(lambda x, pos: "%.3f" % x ))
                        cbar.ax.tick_params(labelsize=10)

                        if k < 4:
                            if j == 1 and k == 3:
                                ax.text(2, 33, '$({})$ head [L]'.format(LetterId[m]), fontsize=14,color='white')
                            else:
                                ax.text(2, 33, '$({})\ t={}$ [T]'.format(LetterId[m],(m+1)*2), fontsize=14,color='white')
                            m = m + 1
                        if np.mod(k,4) == 0:
                            if j == 0:
                                ax.set_ylabel(ylabel[l], fontsize=14)
                                l = 1 + l
                            else:
                                ax.set_ylabel(ylabel[l], fontsize=14)
                                l = 1 + l

                plt.savefig(output_dir + '/epoch_{}_output_{}.png'.format(epoch, idx[i]),
                            bbox_inches='tight',dpi=400)
                plt.close(fig)
                print("epoch {}, done with printing sample output {}".format(epoch, idx[i]))

    rmse_test = np.sqrt(loss / n_out_pixels_test)
    r2_score = 1 - loss / y_test_var
    print("epoch: {}, test r2-score:  {:.6f}".format(epoch, r2_score))
    return r2_score, rmse_test


def cal_R2():
    n_test = args.n_test
    y_mean = np.mean(y_test,axis=0)
    nominator = 0.0
    denominator = 0.0
    for i in range(n_test): # compute the mean for each grid at each time instance
        x = x_test[[i]]
        x_tensor = (th.FloatTensor(x)).to(device)
        model.eval()
        with th.no_grad():
            y_hat = model(x_tensor)
        y_hat = y_hat.data.cpu().numpy()
        nominator = nominator + ((y_test[[i]] - y_hat)**2).sum()
        denominator = denominator + ((y_test[[i]] - y_mean)**2).sum()

    R2 = 1 - nominator/denominator
    print("R2: {}".format(R2))
    return R2

# # * * * Uncomment the following lines to test using pretrained model * * * # #
# print('start predicting...')
# # load model
# model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
# print('Loaded model')
# test(200, 25)
# sys.exit(0)

# MAIN ==============
tic = time()
r2_train, r2_test = [], []
rmse_train, rmse_test = [], []
R2_test_self = []
for epoch in range(1, args.n_epochs + 1):
    # train
    model.train()
    mse = 0.
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target= input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        loss = F.l1_loss(output, target,size_average=False)

        # for computing the RMSE criterion solely
        loss_mse = F.mse_loss(output, target,size_average=False)

        loss.backward()
        optimizer.step()
        mse += loss_mse.item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / train_stats['y_var']
        print("epoch: {}, training r2-score: {:.6f}".format(epoch, r2_score))
        r2_train.append(r2_score)
        rmse_train.append(rmse)
        r2_t, rmse_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)

    scheduler.step(rmse)

    # save model
    if epoch == args.n_epochs:
        th.save(model.state_dict(), model_dir + "/model_epoch{}.pth".format(epoch))

tic2 = time()
print("Done training {} epochs with {} data using {} seconds"
      .format(args.n_epochs, args.n_train, tic2 - tic))

x = np.arange(args.log_interval, args.n_epochs + args.log_interval,
                args.log_interval)
plt.figure()
plt.plot(x, r2_train, 'k', label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
plt.plot(x, r2_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('$R^2$', fontsize=14)
plt.legend(loc='lower right')
plt.savefig(exp_dir + "/r2.png", dpi=400)
plt.close()
np.savetxt(exp_dir + "/r2_train.txt", r2_train)
np.savetxt(exp_dir + "/r2_test.txt", r2_test)

plt.figure()
plt.plot(x, rmse_train, 'k', label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
plt.plot(x, rmse_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.legend(loc='upper right')
plt.savefig(exp_dir + "/rmse.png", dpi=400)
plt.close()
np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)

# save args and time taken
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
args_dict['time'] = tic2 - tic
n_params, n_layers = model._num_parameters_convlayers()
args_dict['num_layers'] = n_layers
args_dict['num_params'] = n_params
with open(exp_dir + "/args.txt", 'w') as file:
    file.write(json.dumps(args_dict))

R2_test_s = cal_R2()
R2_test_self.append(R2_test_s)
np.savetxt(exp_dir + "/R2_test_self.txt", R2_test_self)
