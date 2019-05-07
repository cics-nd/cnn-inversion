"""
Convolutional Encoder-Decoder Networks for Image-to-Image Regression

"""

from dense_ed import DenseED
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
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
from data_plot import load_data, plot_pred, plot_r2_rmse
plt.switch_backend('agg')

# default to use cuda
parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='Net', help='experiment name')
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

kwargs = {'num_workers': 4,'pin_memory': True} if th.cuda.is_available() else {}

# load training data
# shape of x and y: N x Nc x H x W, where N: Number of samples, Nc: Number of input/output channels, H x W: image size
hdf5_dir = args.data_dir + "raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_train)
x_train, y_train, n_out_pixels_train, y_train_var, train_loader = load_data(hdf5_dir, args, kwargs, 'train')
# load test data
hdf5_dir = args.data_dir + "raw2/lsx4_lsy2_var0.5_smax8_D1r0.1/kle{}_lhs{}".format(args.kle_terms,args.n_test)
x_test, y_test, n_out_pixels_test, y_test_var, test_loader = load_data(hdf5_dir, args, kwargs, 'test')

model = DenseED(x_train.shape[1], y_train.shape[1], blocks=args.blocks, growth_rate=args.growth_rate,
                drop_rate=args.drop_rate, bn_size=args.bn_size,
                num_init_features=args.init_features, bottleneck=args.bottleneck).to(device)
print(model)
print("number of parameters: {}\nnumber of layers: {}".format(*model._num_parameters_convlayers()))

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                    verbose=True, threshold=0.0001, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-08)

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
                plot_pred(samples_target, samples_output, epoch, idx[i], output_dir)

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

# # * * * Uncomment the following lines to test using the pretrained model * * * # #
# print('start predicting...')
# # load model
# model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
# print('Loaded model')
# test(200, 25)
# sys.exit(0)

# network training ==============
print("Start training network")
tic = time()
r2_train, r2_test, rmse_train, rmse_test = [], [], [], []
for epoch in range(1, args.n_epochs + 1):
    model.train()
    mse = 0.
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target= input.to(device), target.to(device)
        model.zero_grad()
        output = model(input)
        loss = F.l1_loss(output, target,size_average=False)
        loss.backward()
        optimizer.step()
        mse += F.mse_loss(output, target,size_average=False).item()

    rmse = np.sqrt(mse / n_out_pixels_train)
    scheduler.step(rmse)

    if epoch % args.log_interval == 0:
        r2_score = 1 - mse / y_train_var
        print("epoch: {}, training r2-score: {:.4f}".format(epoch, r2_score))
        r2_train.append(r2_score)
        rmse_train.append(rmse)
        r2_t, rmse_t = test(epoch, plot_intv=args.plot_interval)
        r2_test.append(r2_t)
        rmse_test.append(rmse_t)

    # save model
    if epoch == args.n_epochs:
        th.save(model.state_dict(), model_dir + "/model_epoch{}.pth".format(epoch))

tic2 = time()
print("Done training {} epochs with {} data using {} seconds"
      .format(args.n_epochs, args.n_train, tic2 - tic))

# plot the convergence of R2 and RMSE
plot_r2_rmse(r2_train, r2_test, rmse_train, rmse_test, exp_dir, args)

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
R2_test_self = []
R2_test_self.append(R2_test_s)
np.savetxt(exp_dir + "/R2_test_self.txt", R2_test_self)
