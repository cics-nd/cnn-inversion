clear all
% Basic settings
% Information about model parameters
Par.N_Iter = 20;  % number of iterations
Par.Ne = 6000;    % ensemble size of ES
Par.alpha = 0.1;  % a scalar within [0 1]

Nkle = 679;
% Information about measurements
range = [-4 * ones(1,Nkle), 3, 4, 0*ones(1,5);
          4 * ones(1,Nkle), 5, 6, 8*ones(1,5)];
Par.range = range';
meas = load('obs_sd.dat');
Par.obs = meas(:,1); # observations
Par.sd  = meas(:,2); # standard deviations of the observation error
Par.Nobs = size(meas,1);
save Par.mat Par