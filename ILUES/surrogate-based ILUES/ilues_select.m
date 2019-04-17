function [] = ilues_select()

Parallel_Computing = 0; % 0 for serial computing, 1 for parallel computing
if Parallel_Computing == 1
    Ncpu = 12;
    myCluster = parcluster('local'); % reset the maximum number of workers (i.e., cores)
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster);
    N = maxNumCompThreads;
    LASTN = maxNumCompThreads(N);
    LASTN = maxNumCompThreads('automatic');
    parpool('local',Ncpu); % start parpool with Ncpu cores, the default profile is 'local'
end

load Par.mat  # load the parameter settings used in ILUES
datax = load('xf.mat');
xf = datax.xf;
datay = load('yf.mat');
yf = datay.yf;
load x1.mat

obs = Par.obs;
Nobs = Par.Nobs;     % number of the measurements
Ne = Par.Ne;        % ensemble size
sd = Par.sd;
range = Par.range;
alpha = Par.alpha;
N_Iter = Par.N_Iter;
beta = sqrt(N_Iter);

Cd = eye(Nobs);         
for i = 1:Nobs
    Cd(i,i) = sd(i)^2;  % covariance of the measurement errors   
end

meanxf = repmat(mean(x1,2),1,Ne);           % mean of the prior parameters
Cm = (x1 - meanxf)*(x1 - meanxf)'/(Ne - 1); % auto-covariance of the prior parameters
      
J1 = nan(Ne,1);
for i = 1:Ne
    J1(i,1) = (yf(:,i)-obs)'/Cd*(yf(:,i)-obs);
end

xa = nan(size(x1));  % define the updated ensemble    
parfor j = 1:Ne
    xa(:,j) = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,j);
end
save xa.mat xa
	
end 


function xest = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,jj)

% The local updating scheme used in ILUES

Ne = size(xf,2);
xr = xf(:,jj);

xr = repmat(xr,1,Ne);
J = (xf - xr)'/Cm*(xf - xr);
J2 = diag(J);

J3 = J1/max(J1) + J2/max(J2);
M = ceil(Ne*alpha);


[J3min,index] = min(J3);
xl = xf(:,index);
yl = yf(:,index);
alfa = J3min ./ J3;
alfa(index) = 0;
index1 = RouletteWheelSelection(alfa,M-1);
xl1 = xf(:,index1);
yl1 = yf(:,index1);
xl = [xl,xl1];
yl = [yl,yl1];

xu =  updatapara(xl,yl,range,sd*beta,obs);
% xest = xu(:,randperm(M,1));
a=clock; rng(floor(sum(a(:))*200));
xest = xu(:,randperm(M,1));
% xest = xu(:,1);

end


function xa = updatapara(xf,yf,range,sd,obs)

% Update the model parameters via the ensemble smoother

Npar = size(xf,1);
Ne = size(xf,2);
Nobs = length(obs);

Cd = eye(Nobs);
for i = 1:Nobs
    Cd(i,i) = sd(i)^2;
end

meanxf = repmat(mean(xf,2),1,Ne);
meanyf = repmat(mean(yf,2),1,Ne);
Cxy =  (xf - meanxf)*(yf - meanyf)'/(Ne - 1);
Cyy =  (yf - meanyf)*(yf - meanyf)'/(Ne - 1);

kgain = Cxy/(Cyy + Cd);
obse = repmat(obs,1,Ne) + normrnd(zeros(Nobs,Ne),repmat(sd,1,Ne));
xa = xf + kgain*(obse - yf);

% Boundary handling
for i = 1:Ne
    for j = 1:Npar
        if xa(j,i) > range(j,2)
            xa(j,i) = (range(j,2) + xf(j,i))/2;
        elseif xa(j,i) < range(j,1)
            xa(j,i) = (range(j,1) + xf(j,i))/2;
        end
    end
end

end

