clear all;clc;close all

% gcp;                               % open parallel computation
Parallel_Computing = 1; % 0 for serial computing, 1 for parallel computing
if Parallel_Computing == 1
    Ncpu = 8;
    myCluster = parcluster('local'); % reset the maximum number of workers (i.e., cores)
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster);
    N = maxNumCompThreads;
    LASTN = maxNumCompThreads(N);
    LASTN = maxNumCompThreads('automatic');
    parpool('local',Ncpu); % start parpool with Ncpu cores, the default profile is 'local'
end

% Basic settings
Ne = 6000;                         % ensemble size of ES
alpha = 0.1;                       % a scalar within [0 1]
N_Iter = 20;                       % number of iterations

% Information about model parameters
Npar = 686;                        % number of unknown parameters
Nkle = 679;
Par.Nkle = Nkle;
Par.MeanY = 2.0;
load eig_vals.mat
load eig_vecs.mat
Par.eig_vals = eig_vals;
Par.eig_vecs = eig_vecs;

% Information about measurements
range = [-4 * ones(1,Nkle), 3, 4, 0*ones(1,5);
          4 * ones(1,Nkle), 5, 6, 8*ones(1,5)];
range = range';
meas = load('obs_sd.dat');
obs = meas(:,1);
sd  = meas(:,2);
Nobs = size(meas,1);

% Copy files for parallel computation
copyexample(Ne);

% Draw Ne samples from the prior distribution
flag = 0; % load existing data? 1: load initial samples;  2: restart; 0: No
if flag == 1
    load x1.mat ;
    load y1.mat;
elseif flag == 2
    load xa17.mat 
    load ya17.mat
    x1 = xa;
    y1 = ya;
else
    x1 = genex(range,Ne,Npar,Nkle);
    save x1.mat x1
    y1 = nan(Nobs,Ne);
    kle_coefs = x1(1:Nkle, :);
    log_K = Par.MeanY + Par.eig_vecs * sqrt(Par.eig_vals) * kle_coefs;
    K = exp(log_K);
    source = x1(Nkle+1:Npar, :);
    parfor i = 1:Ne
        cond = reshape(K(:, i), 41, 81);
        y1(:,i) = forward_model(cond,source(:,i),i);
    end
    save y1.mat y1
end

% Update the parameters with ILUES
[xall,yall] = ilues(x1,y1,range,obs,sd,alpha,Par,N_Iter);

% save -v7.3 results.mat;
copyexample(Ne,-1);

delete(gcp('nocreat')); % close the parpool



