import numpy as np
import time
import sys
import os
import torch as th
import scipy.io
from attrdict import AttrDict

from dense_ed import DenseED
import argparse
from torch.utils.data import DataLoader

import matlab.engine
eng = matlab.engine.start_matlab()

parser = argparse.ArgumentParser(description='Dnense Encoder-Decoder Convolutional Network')
parser.add_argument('--exp-name', type=str, default='single', help='experiment name')
parser.add_argument('--skip', action='store_true', default=False, help='enable skip connection between encoder and decoder nets')
parser.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
parser.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
parser.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
parser.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
parser.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
parser.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')
parser.add_argument('--data-dir', type=str, default="/afs/crc.nd.edu/user/s/smo/invers_mt3d/DCEDN_sequence_2out/", help='data directory')
parser.add_argument('--kle-terms', type=int, default=679, help='num of KLE terms')
parser.add_argument('--n-train', type=int, default=1500, help="number of training data")
parser.add_argument('--n-test', type=int, default=500, help="number of test data")
parser.add_argument('--loss-fn', type=str, default='l1', help='loss function: mse, l1, huber, berhu')
parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005, help='learnign rate')
parser.add_argument('--weight-decay', type=float, default=5e-5, help="weight decay")
parser.add_argument('--batch-size', type=int, default=200, help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=50, help='input batch size for testing (default: 100)')
parser.add_argument('--log-interval', type=int, default=5, help='how many epochs to wait before logging training status')
parser.add_argument('--plot-interval', type=int, default=50, help='how many epochs to wait before plotting training status')

args = parser.parse_args()
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model = DenseED(3, 2, blocks=args.blocks, growth_rate=args.growth_rate,
                        drop_rate=args.drop_rate, bn_size=args.bn_size,
                        num_init_features=args.init_features, bottleneck=args.bottleneck).to(device)

Par=AttrDict()

# Load pretrained model
model_dir = "/afs/crc.nd.edu/user/s/smo/invers_mt3d/DCEDN_sequence_2out/experiments/Oct_29_smax8_var0.5_D1r0.1/single/kle_679/Ntrs1500__Bks(5, 10, 5)_Bts200_Eps200_wd5e-05_lr0.005_K40_2loss_l1-w5/"
model.load_state_dict(th.load(model_dir + '/model_epoch{}.pth'.format(args.n_epochs)))
print('Loaded model')

device = th.device("cuda" if th.cuda.is_available() else "cpu") # training on GPU or CPU

def run_surrogate(X, Par, model):

    model.eval()
    cond, source = gen_input4net(X, Par)
    x = np.full((1,3,41,81), 0.0)           # three input channels: hydraulic conductivity field, source term, previous concentration field
    y = np.full( (Par.Nt,2,41,81), 0.0) # two output channles: concentration and head fields
    y_i_1 = np.full((41,81), 0.0)   # y_0 = 0
    for i in range(Par.Nt):
        x[0,0,:,:] = cond           # hydraulic conductivity
        x[0,1,:,:] = source[i]      # source rate
        x[0,2,:,:] = y_i_1          # the i-1)^th predicted concentration field, which is treated as an input channel
        x_tensor = (th.FloatTensor(x)).to(device)
        with th.no_grad():
            y_hat = model(x_tensor)
        y_hat = y_hat.data.cpu().numpy()
        y[i] = y_hat
        y_i_1 = y_hat[0,0,:,:]      # the updated (i-1)^th predicted concentration field

    y_pred = np.full( (Par.Nt + 1,41,81), 0.0)
    y_pred[:Par.Nt] = y[:,0]   # the concentration fields at Nt time instances
    y_pred[Par.Nt]  = y[0,1] # the hydraulic head field

    source_loc = X[Par.Nkle : Par.Nkle+2]
    x_coord, y_coord = get_coordinate(source_loc) # get the coordinate for interpolation, it changes with the source location
    y_sim_obs = get_simv(x_coord, y_coord, y_pred)  # get the simulated outputs at observation locations using interpolation

    return y_sim_obs

def gen_input4net(X, Par):

    KLE_coefs = X[:Par.Nkle]
    source_loc = X[Par.Nkle : Par.Nkle+2]
    Sx_id = int( np.floor(source_loc[0]/0.25) )  # source location id in the x-axis
    Sy_id = int( np.floor(source_loc[1]/0.25) )  # source location id in the y-axis
    source_rate = X[Par.Nkle+2 : Par.Npar]
    log_K = Par.MeanY + np.dot(np.dot(Par.eig_vecs, np.sqrt(Par.eig_vals)), np.transpose(KLE_coefs) )
    cond = np.transpose( np.reshape(log_K, (81, 41)) )
    source = np.full( (Par.Nt,41,81), 0.0)
    for j in range(Par.Nt-2):
        source[j, Sy_id, Sx_id] = source_rate[j]

    return cond, source

def get_coordinate(source_loc):

    sx = source_loc[0]
    sy = source_loc[1]

    dx1, dx2, nx1 = discretize(20,sx)
    xx = np.full((81), 0.0)
    xx[:nx1] = dx1
    xx[nx1:] = dx2
    xx = np.cumsum(xx)
    xx[:nx1] = xx[:nx1] - dx1/2
    xx[nx1:] = xx[nx1:] - dx2/2

    dy1, dy2, ny1 = discretize(10,sy)
    yy = np.full((41), 0.0)
    ny2 = 41 - ny1
    yy[:ny2] = dy2  # this is different from that in x-direction, to be consistent with the output file of PM.exe
    yy[ny2:] = dy1
    yy = np.cumsum(yy)
    yy[:ny2] = yy[:ny2] - dy2/2
    yy[ny2:] = yy[ny2:] - dy1/2

    x = np.full((3321,1), 0.0)
    y = np.full((3321,1), 0.0)
    k = 0
    for i in range(41):
        for j in range(81):
            x[k,0] = xx[j]
            y[k,0] = yy[40 - i]
            k = k + 1

    x = np.reshape(x, (41, 81))
    y = np.reshape(y, (41, 81))

    return x, y

def get_simv(x, y, y_pred):

    # # One can also use the built-in Rbf interpolation of Python,
    # # but here we use the interpolation function in MATLAB instead, as it is about 10 times faster
    # f = Rbf(x, y, y_output,  function='multiquadric')  
    # y_sim = f(xobs, yobs)
    scipy.io.savemat('InterpData.mat', dict(x=x, y=y, y_pred=y_pred))
    eng.interp_matlab(nargout=0)
    y_sim = np.loadtxt("y_sim.dat")

    return y_sim

def discretize(L,loc):

    N = round(L / 0.25)
    n1 = int(np.floor(loc / 0.25) + 1)
    dx1 = loc / (n1 - 1)
    n2 = N + 1 - n1
    dx2 = (L - (n1 - 0.5) * dx1) / (n2 - 0.5)

    return dx1, dx2, n1	

# load the eigen vectors and eigen values of the KLE expansion	
eig_vecs = scipy.io.loadmat('eig_vecs.mat')
eig_vecs = eig_vecs['eig_vecs']
eig_vals = scipy.io.loadmat('eig_vals.mat')
eig_vals = eig_vals['eig_vals']
Par.eig_vecs = eig_vecs
Par.eig_vals = eig_vals


Par.Npar = 686
Par.Nkle = 679
Par.Nt = 7
Par.Nobs = 168
Par.MeanY = 2
N_iter = 20

flag = 0  # load exiting data? 1: Yes; 0: No
if flag == 1:
    x1 = scipy.io.loadmat('x1.mat')
    xf = x1['x1']
    y1 = scipy.io.loadmat('y1.mat')
    yf = y1['y1']
else:
    x1 = scipy.io.loadmat('x1.mat')
    x1 = x1['x1']     # input matrix: Npar x Ne, to be consistent with the matrix structure used in ILUES
    x1 = x1.T         # Matrix transposition, to be consistent with the matrix structure used in DNN
    y1 = np.zeros((x1.shape[0], Par.Nobs))
    for i in range(x1.shape[0]):  # (x1, y1) are initial samples
        y1[i,:] = run_surrogate(x1[i,:], Par, model)
    y1 = y1.T
    scipy.io.savemat('y1.mat', dict(y1=y1))
    xf = x1.T
    yf = y1

scipy.io.savemat('xf.mat', dict(xf=xf))  # Initial ensemble
scipy.io.savemat('yf.mat', dict(yf=yf))
xall = xf
yall = yf
for i in range(N_iter):
    eng.ilues_select(nargout=0)  # Update the ensemble using ILUES
    xa = scipy.io.loadmat('xa.mat')  # The candidates for update
    xa = xa['xa']
    xa = xa.T
    ya = np.zeros((xa.shape[0], Par.Nobs))
    for j in range(xa.shape[0]):
        ya[j,:] = run_surrogate(xa[j,:], Par, model)
    ya = ya.T
    scipy.io.savemat('ya.mat', dict(ya=ya))  # The predicted outputs at candidate points
    eng.update_samples(nargout=0)  # accept or reject the candidate?
    xa = scipy.io.loadmat('xa.mat')  # The updated inputs
    ya = scipy.io.loadmat('ya.mat')  # The updated outputs
    xa = xa['xa']
    ya = ya['ya']
    xall = np.concatenate((xall,xa),axis=1)
    yall = np.concatenate((yall,ya),axis=1)
    scipy.io.savemat('results.mat', dict(xall=xall,yall=yall))  # save results

