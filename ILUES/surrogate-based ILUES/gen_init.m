clear all
# generate the initial input ensemble
Ne = 6000;
Npar = 686;                        % number of unknown parameters
Nkle = 679;
range = [-4 * ones(1,Nkle), 3, 4, 0*ones(1,5);
          4 * ones(1,Nkle), 5, 6, 8*ones(1,5)];
range = range';
x1 = genex(range,Ne,Npar,Nkle);
save x1.mat x1