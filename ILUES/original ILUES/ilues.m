function [xall,yall] = ilues(xf,yf,range,obs,sd,alpha,Par,N_Iter)

Nobs = length(obs);     % number of the measurements
Npar = size(xf,1);      % number of the parameters
Ne = size(xf,2);        % ensemble size

Cd = eye(Nobs);         
for i = 1:Nobs
    Cd(i,i) = sd(i)^2;  % covariance of the measurement errors   
end
factor = ones(N_Iter,1)*sqrt(N_Iter);       % inflate Cd in the multiple data assimilation scheme
% factor = ones(N_Iter,1)*sqrt(37);       % inflate Cd in the multiple data assimilation scheme

xall = xf; yall = yf;                       % store results at each iteration

meanxf = repmat(mean(xf,2),1,Ne);           % mean of the prior parameters
Cm = (xf - meanxf)*(xf - meanxf)'/(Ne - 1); % auto-covariance of the prior parameters

for n_i = 1 : N_Iter
    xa = nan(size(xf)); ya = nan(size(yf));     % define the updated ensemble
      
    J1 = nan(Ne,1);
    for i = 1:Ne
        J1(i,1) = (yf(:,i)-obs)'/Cd*(yf(:,i)-obs);
    end
    
    beta = factor(n_i);
    tic;
    parfor j = 1:Ne
        xa(:,j) = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,j);
    end
    toc;
    
    kle_coefs = xa(1:Par.Nkle, :);
    log_K = Par.MeanY + Par.eig_vecs * sqrt(Par.eig_vals) * kle_coefs;
    K = exp(log_K);
    source = xa(Par.Nkle+1:Npar, :);
    parfor i = 1:Ne
        cond = reshape(K(:, i), 41, 81);
        ya(:,i) = forward_model(cond,source(:,i),i);
    end
    
    % Enhance the performance of ILUES in problems with many parameters 
    if Npar > 10    
        likf = Cal_Log_Lik(yf,obs,sd);   
        lika = Cal_Log_Lik(ya,obs,sd);
        a=clock; rng(floor(sum(a(:))*20));
        cc = (exp(lika - likf)) < rand(Ne,1);
        xa(:,cc) = xf(:,cc);
        ya(:,cc) = yf(:,cc);
    end
    

    
    xf = xa; yf = ya;
    
 
    save(['xa',num2str(n_i),'.mat'],'xa')
    save(['ya',num2str(n_i),'.mat'],'ya')
    
end

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
a=clock; rng(floor(sum(a(:))*10));
xest = xu(:,randperm(M,1));

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
a=clock; rng(floor(sum(a(:))*10));
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

function Lik = Cal_Log_Lik(y1,obs,sd)

% Log-transformed Gaussian likelihood function

[~,N] = size(y1);
Lik = zeros(N,1);

for i = 1:N
    Err = obs - y1(:,i);
    Lik(i) = - ( length(Err) / 2) * log(2 * pi) - sum ( log( sd ) ) - ...
        1/2 * sum ( ( Err./sd ).^2);
end

end

