import numpy as np
import h5py
from scipy.stats import variation


def read_output(ndata, ntimes, Nt, ngx, ngy):
    y = np.full( (ndata,ntimes,ngx,ngy), 0.0)
    for i in range(1, ndata + 1):
        for j in range(1, ntimes + 1):
            y[i-1, j-1, :, :] = np.loadtxt("output/conc_{}_t_{}.dat".format(i,j))

    y = np.where(y>0.0,y,0.0)
    y0 = y[:,:Nt]
    # id0 = y0.nonzero()
    # y0 = y0[id0]

    y1 = y[:,Nt:]
    # id1 = y1.nonzero()
    # y1 = y1[id1]

    # y0_mean = np.average(y0)
    # y1_mean = np.average(y1)
    y_cov0 = variation(y0,axis=None)
    y_cov1 = variation(y1,axis=None)
    # print("cov0: {}".format(y_cov0))
    # print("cov1: {}".format(y_cov1))
    weight = y_cov0 / y_cov1
    weight = 5
    print("weight:{}".format(weight))
    with open("weight.txt", "w") as text_file:
        text_file.write("%f" % weight)
    return weight


def read_input_sourceIndex_2output(ndata, ntimes, Nt, ngx, ngy):
    # ntimes is the total number of time instances considered
    # Nt is the number of time instances with non-zero source rate

    x = np.full( (ndata*ntimes,3,ngx,ngy), 0.0)         # three input channels: (K, r_i, y_{i-1})
    y = np.full( (ndata*ntimes,2,ngx,ngy), 0.0)         # one output channels: y_i
    k = 0
    for i in range(1,ndata+1):
        K = np.loadtxt("input/cond{}.dat".format(i))  # hydraulic conductivity
        K = np.log(K)
        source = np.loadtxt("input/ss{}.dat".format(i))
        head = np.loadtxt("output/head_{}.dat".format(i))

        Sx_id = int( np.floor(source[0]/0.25) )  # source location id in the x-axis
        Sy_id = int( np.floor(source[1]/0.25) )  # source location id in the y-axis

        S_rate = source[2:]
        y_j_1 = np.full( (ngx,ngy), 0.0)                # y_0 = 0
        for j in range(1,ntimes+1):
            x[k,0,:,:] = K                              # K is the first input channel
            if j <= Nt:
                x[k,1,Sy_id, Sx_id] = S_rate[j-1]       # source rate is the second input channel
            x[k,2,:,:] = y_j_1                          # the (j-1)th output is the third output channel

            y_j = np.loadtxt("output/conc_{}_t_{}.dat".format(i,j))
            y_j = np.where(y_j>0, y_j, 0.0)
            y[k,0,:,:] = y_j                            # the jth output
            y[k,1,:,:] = head                           # head is the second output channel
            y_j_1 = y_j
            k += 1

    print("x: {}".format(x[:ntimes,:]))
    print("y_c: {}".format(y[:ntimes,0]))
    print("y_h: {}".format(y[ntimes,1]))
    print("Ymax: {}".format(np.max(y[:,:,:,:])))

    hf = h5py.File('input_lhs{}_2out.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()

    hf = h5py.File('output_lhs{}_2out.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = y, dtype ='f', compression = 'gzip')
    hf.close()


ndata = 1500
ntimes = 7
Nt = 5  # the number of time instances with non-zero source rate
ngx = 41
ngy = 81
read_input_sourceIndex_2output(ndata, ntimes, Nt, ngx, ngy)

