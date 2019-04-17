"""
Autoregressive Convolutional Encoder-Decoder Networks for Image-to-Image Regression

"""

from dense_ed import DenseED
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
parser.add_argument('--exp-name', type=str, default='AR-Net', help='experiment name')
parser.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

parser.add_argument('--data-dir', type=str, default="/afs/crc.nd.edu/user/s/smo/invers_mt3d/DCEDN_sequence_2out/", help='data directory')
parser.add_argument('--kle-terms', type=int, default=679, help='num of KLE terms')
parser.add_argument('--n-train', type=int, default=400, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")

parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005, help='learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=200, help='input batch size for training (default: 100)')
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

exp_dir = args.data_dir + all_over_again + "/{}/kle_{}/Ntrs{}__Bks{}_Bts{}_Eps{}_wd{}_lr{}_K{}_w_c{}".\
    format(args.exp_name, args.kle_terms, args.n_train,args.blocks,
           args.batch_size, args.n_epochs, args.weight_decay, args.lr, args.growth_rate,args.w_c)

output_dir = exp_dir + "/predictions"
model_dir = exp_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

ntimes = 7  # ntimes is the total number of time instances considered
Nt = 5      # Nt is the number of time instances with non-zero source rate

# load training data
hdf5_dir = "/afs/crc.nd.edu/user/s/smo/invers_mt3d/raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_train)
with h5py.File(hdf5_dir + "/input_lhs{}_2out.hdf5".format(args.n_train), 'r') as f:
    x_train = f['dataset'][()]
    print("total input data shape: {}".format(x_train.shape))
with h5py.File(hdf5_dir + "/output_lhs{}_2out.hdf5".format(args.n_train), 'r') as f:
    y_train = f['dataset'][()]
    print("total output data shape: {}".format(y_train.shape))
# load test data
hdf5_dir = "/afs/crc.nd.edu/user/s/smo/invers_mt3d/raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_test)
with h5py.File(hdf5_dir + "/input_lhs{}_2out.hdf5".format(args.n_test), 'r') as f:
    x_test = f['dataset'][()]
    print("test input data shape: {}".format(x_test.shape))
with h5py.File(hdf5_dir + "/output_lhs{}_2out.hdf5".format(args.n_test), 'r') as f:
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


# compute the quality metrics based on the test data and plot predictions (at specified epochs)
def test(epoch, plot_intv):
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

            print("Index of data: {}".format(idx))
            print("X shape: {}".format(x_test.shape))

            for i in range(n_samples):
                model.eval()
                x = x_test[ idx[i] * ntimes: (idx[i]+1) * ntimes]
                y = y_test[ idx[i] * ntimes: (idx[i]+1) * ntimes]

                y_output = np.full( (ntimes,y_test.shape[1],41,81), 0.0)
                x_ii = np.full((1,x_test.shape[1],41,81), 0.0)
                y_ii_1 = np.full((41,81), 0.0)     # y_0 = 0
                for ii in range(ntimes):
                    x_ii[0,0,:,:] = x[ii,0,:,:]   # hydraulic conductivity
                    x_ii[0,1,:,:] = x[ii,1,:,:]   # source rate
                    x_ii[0,2,:,:] = y_ii_1        # the ii_th predicted output
                    x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
                    with th.no_grad():
                        y_hat = model(x_ii_tensor)
                    y_hat = y_hat.data.cpu().numpy()
                    y_output[ii] = y_hat
                    y_ii_1 = y_hat[0,0,:,:]  # treat the current output as input to predict the ouput at next time step

                y_target = np.full( (ntimes + 1,1,41,81), 0.0)
                y_target[:ntimes] = y[:,[0]]  # the concentration fields for one input conductivity fields at ntimes time steps
                y_target[ntimes] = y[[0],[1]] # the pressure field
                y_pred = np.full( (ntimes + 1,1,41,81), 0.0)
                y_pred[:ntimes] = y_output[:,[0]]
                y_pred[ntimes] = y_output[[0],[1]]

                samples = np.vstack((y_target, y_pred, y_target - y_pred))

                column = ntimes + 1
                c_max = np.full( (column*3), 0.0) # the same color scale for the predicted output fields at the same time step
                for l in range(column*3):
                    if l < column:
                        c_max[l] = np.max(samples[l])
                    elif column <= l < 2*column:
                        c_max[l] = np.max(samples[l])
                        if c_max[l] > c_max[l-column]:
                            c_max[l-column] = c_max[l]
                        else:
                            c_max[l] = c_max[l-column]
                    else:
                        c_max[l] = np.max( np.abs(samples[l]) )

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
                            cax = ax.contourf(samples[ s_id,0], np.arange(0.0 , c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet')
                            fig.add_subplot(ax)
                        else:
                            cax = ax.contourf(samples[ s_id,0], np.arange(0.0 - c_max[s_id] - c_max[s_id]/nl*1, c_max[s_id] + c_max[s_id]/nl*1, c_max[s_id]/nl), cmap='jet',extend='both')
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
    r2_score = 1 - loss / test_stats['y_var']
    print("epoch: {}, test r2-score:  {:.4f}".format(epoch, r2_score))
    return r2_score, rmse_test

def cal_R2():
    "compute the test R2 score"
    n_test = args.n_test
    y_sum = np.full( (ntimes + 1,1,41,81), 0.0)

    for i in range(n_test):
        y = np.full( (ntimes + 1,1,41,81), 0.0)
        y[:ntimes] = y_test[i * ntimes: (i+1) * ntimes,[0]] # concentration at n_t time instances
        y[ntimes] = y_test[ i * ntimes,[1] ] # head
        y_sum = y_sum + y
    y_mean = y_sum / n_test

    nominator = 0.0
    denominator = 0.0
    for i in range(n_test):
        x = x_test[ i * ntimes: (i+1) * ntimes]
        y = np.full( (ntimes + 1,1,41,81), 0.0)
        y[:ntimes] = y_test[i * ntimes: (i+1) * ntimes,[0]] # concentration at n_t time instances
        y[ntimes] = y_test[ i * ntimes,[1] ] # head

        y_output = np.full( (ntimes + 1, 1,41,81), 0.0)
        x_ii = np.full((1,x_test.shape[1],41,81), 0.0)
        y_ii_1 = np.full((41,81), 0.0)     # y_0 = 0
        for ii in range(ntimes):
            x_ii[0,0,:,:] = x[ii,0,:,:]   # hydraulic conductivity
            x_ii[0,1,:,:] = x[ii,1,:,:]   # source rate
            x_ii[0,2,:,:] = y_ii_1        # the ii_th predicted output
            x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
            model.eval()
            with th.no_grad():
                y_hat = model(x_ii_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y_output[ii,0] = y_hat[0,0]
            if ii == ntimes - 1:
                y_output[ii+1,0] = y_hat[0,1]
            y_ii_1 = y_hat[0,0,:,:]
        nominator = nominator + ((y - y_output)**2).sum()
        denominator = denominator + ((y - y_mean)**2).sum()

    R2 = 1 - nominator/denominator
    print("R2: {}".format(R2))
    return R2

# find the maximum absolute prediction error at Nt concentration fields in each test sample,
# i.e., the results shown in Figure 13 of the paper
def max_err():
    n_test = args.n_test
    ErrMax = np.zeros((n_test*Nt))
    for i in range(n_test):
        x = x_test[ i * ntimes: (i+1) * ntimes - 2]
        y = np.full( (Nt,1,41,81), 0.0)
        y[:Nt] = y_test[i * ntimes: (i+1) * ntimes - 2,[0]] # concentration at n_t time instances

        y_output = np.full( (Nt, 1,41,81), 0.0)
        x_ii = np.full((1,x_test.shape[1],41,81), 0.0)
        y_ii_1 = np.full((41,81), 0.0)     # y_0 = 0
        for ii in range(Nt):
            x_ii[0,0,:,:] = x[ii,0,:,:]   # hydraulic conductivity
            x_ii[0,1,:,:] = x[ii,1,:,:]   # source rate
            x_ii[0,2,:,:] = y_ii_1        # the ii_th predicted output
            x_ii_tensor = (th.FloatTensor(x_ii)).to(device)
            model.eval()
            with th.no_grad():
                y_hat = model(x_ii_tensor)
            y_hat = y_hat.data.cpu().numpy()
            y_output[ii,0] = y_hat[0,0]
            y_ii_1 = y_hat[0,0,:,:]
        err = np.abs(y - y_output)

        ErrMax[i*Nt : (i+1)*Nt] = ( ( err.max(axis=1) ).max(axis=1) ).max(axis=1)

    np.savetxt(exp_dir +'/TestErrMax_ntrain{}.dat'.format(args.n_train), ErrMax, fmt='%10.4f')   # use exponential notation
    return None


# # * * * Uncomment the following lines to test using pretrained model * * * # #
# print('start predicting...')
# # load model
# model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
# print('Loaded model')
# test(200, 25)
# sys.exit(0)

# MAIN ==============
tic = time()
R2_test_self = []
r2_train, r2_test = [], []
rmse_train, rmse_test = [], []
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
        print("loss: {}".format(loss))

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

max_err()
