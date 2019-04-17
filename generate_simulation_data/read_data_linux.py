import numpy as np
import h5py
def read_input(ndata, dx, ngx, ngy):
    x = np.full( (ndata,dx,ngx,ngy), 0.0)
    for i in range(1, ndata + 1):
        K = np.loadtxt("input/cond{}.dat".format(i))
        x[i-1,0, :, :] = np.log(K)  # K is the first input channel, log transformed
        source = np.loadtxt("input/ss{}.dat".format(i))
        Sx_id = int( np.floor(source[0]/0.25) )  # source location id in the x-axis
        Sy_id = int( np.floor(source[1]/0.25) )  # source location id in the y-axis
        S_rate = source[2:]
        for j in range(1,dx):
            x[i-1,j,Sy_id, Sx_id] = S_rate[j-1]       # n_t source terms ara treated as n_t input channels

    print("X: {}".format(x[0,]))
    hf = h5py.File('input_lhs{}.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()

def read_output(ndata, dy, ngx, ngy):
    y = np.full( (ndata,dy,ngx,ngy), 0.0)
    for i in range(1, ndata + 1):
        for j in range(1, dy):
            y[i-1, j-1, :, :] = np.loadtxt("output/conc_{}_t_{}.dat".format(i,j))
        y[i-1, dy-1, :, :] = np.loadtxt("output/head_{}.dat".format(i)) 

    y = np.where(y>0,y,0.0)
    print("Y: {}".format(y[0,]))
    hf = h5py.File('output_lhs{}.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = y, dtype ='f', compression = 'gzip')
    hf.close()

ndata = 1000
dx = 6
dy = 8 
ngx = 41
ngy = 81
read_input(ndata,dx,ngx,ngy)
read_output(ndata,dy,ngx,ngy)
