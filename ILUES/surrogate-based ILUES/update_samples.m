function [] = update_samples()

load Par.mat
load xf.mat
load yf.mat

obs = Par.obs;
Ne = Par.Ne;        % ensemble size
sd = Par.sd;

load xa.mat
load ya.mat  % predicted outputs at suggested updated inputs

likf = Cal_Log_Lik(yf,obs,sd);   
lika = Cal_Log_Lik(ya,obs,sd);
cc = (exp(lika - likf)) < rand(Ne,1);
xa(:,cc) = xf(:,cc);
ya(:,cc) = yf(:,cc);

xf = xa; yf = ya;
save xf.mat xf
save yf.mat yf
save xa.mat xa
save ya.mat ya

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