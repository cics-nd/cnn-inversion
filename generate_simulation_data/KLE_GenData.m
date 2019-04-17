
% clear all
kle_num = 679;
n_data = [1500,1000,500,400];

doe = 'lhs';
% doe = 'mc';

seeds = n_data;
if strcmp(doe, 'mc')
    seeds = n_data + 1011;
end

% 1: abs
% 2: exp
% 3: rbf
kernel = 2;
% unit size of domain
sx = 20; sy = 10;

% resolution
% global ngx ngy 
ngx = 81; ngy = 41;

n_grids = ngx * ngy;

% number of KL terms that are preserved
% kle_num = 100;
% percentage to be preserved
kle_percentage = 1.0;
% set the kle terms by kle_percentage ('0') or kle_num ('1')
trunc_with_num = 1;

% Correlation lengths
ls_x = 4;
ls_y = 2;

% Field mean and variance
MeanY = 2;
VarY = 0.5;

% Correlation function
C = nan(n_grids, n_grids);

x = linspace(0, sx, ngx);
y = linspace(0, sy, ngy);
[X, Y] = meshgrid(x, y);
grids = [X(:), Y(:)];

tic
% too low... 
if kernel == 1
    % abs
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY*exp(-(abs((grids(i,1)-grids(j,1))/ls_x)...
                +abs((grids(i,2)-grids(j,2))/ls_y)));
        end
    end
    
elseif kernel == 2
    % exp
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY * exp(- sqrt(...
                ((grids(i, 1) - grids(j, 1)) / ls_x)^2 + ...
                ((grids(i, 2) - grids(j, 2)) / ls_y)^2));
        end
    end
    
elseif kernel == 3
    % rbf
    for i = 1 : n_grids
        for j = 1 : n_grids
            C(i,j) = VarY * exp(- (...
                ((grids(i, 1) - grids(j, 1)) / ls_x)^2 + ...
                ((grids(i, 2) - grids(j, 2)) / ls_y)^2));
        end
    end
end
toc        
% Calculate eigenvalues and eigenvectors

if trunc_with_num
    % eig_vecs: M*N x kle_num
    % eig_vals: kle_num x kle_num
    [eig_vecs, eig_vals] = eigs(C, kle_num);
    ratio = cumsum(diag(eig_vals)) / (n_grids * VarY);
    kle_percentage = ratio(end);
    fprintf('truncated to %d terms preserving %.5f energy, ls = %.2f\n', ...
        kle_num, kle_percentage, min(ls_x, ls_y))
    
    
else
    tic
    [eig_vecs, eig_vals] = eig(C);
    toc
    size(C)
    fprintf('done eig\n')
    eig_vals = eig_vals(end : -1 : 1, end : -1 : 1);
    eig_vecs = eig_vecs(:, end : -1 : 1);
    
    if kle_percentage == 1.0
        fprintf('direct sampling\n')
    else
        ratio = cumsum(diag(eig_vals)) / sum(diag(eig_vals));
        kle_num = find(ratio > kle_percentage, 1);
        eig_vals = eig_vals(1:kle_num, 1:kle_num);
        eig_vecs = eig_vecs(:, 1:kle_num);

        figure
        plot(ratio)
        grid on
        fprintf('truncated to preserve %.5f energy with %d terms, ls = %.2f\n', ...
            kle_percentage, kle_num, min(ls_x, ls_y))
    end
end
toc

save eig_vecs.mat eig_vecs
save eig_vals.mat eig_vals  

source_max = 8;  
for i=1:length(seeds)
    rng(seeds(i))
    input_dir = ['raw2/lsx', num2str(ls_x), '_lsy', num2str(ls_y), '_var', num2str(VarY),'_smax',num2str(source_max),'_D1r0.1',...
                '/kle',num2str(kle_num), '_', doe, num2str(n_data(i)), '/input/'];
   
    if ~exist(input_dir, 'dir')
        mkdir(input_dir);
    end

    % KLE coefficients
    if strcmp(doe, 'lhs')
        % LHS design
        disp(['lhs for ', num2str(n_data(i)), ' data'])
        xi = -1 + 2 * lhsdesign(n_data(i), kle_num);
        kle_terms = sqrt(2) * erfinv(xi);
    elseif strcmp(doe, 'mc')
        disp(['MC for ', num2str(n_data(i)), ' data'])
        % MC sampling
        kle_terms = randn(n_data(i), kle_num);
    end
    
    % n_grids x n_train(i)
    log_K = MeanY + eig_vecs * sqrt(eig_vals) * kle_terms';
    K = exp(log_K);
   
    % save the input into separate files
    for n=1:n_data(i) 
        cond = reshape(K(:, n),ngy,ngx);
        dlmwrite([input_dir, 'cond', num2str(n), '.dat'], cond, 'delimiter', ...
            ' ', 'precision', 8);
    end
    
    % the source location (the first two paras) and the source strength (the remaining paras)
    range = [3 4 zeros(1,5);
             5 6 source_max*ones(1,5)];
    source = lhsdesign( n_data(i), size(range,2) );
    source = RangeTrans(source, range);
    for n=1:n_data(i) 
        ss = source(n,:);
        ss = ss';
        dlmwrite([input_dir, 'ss', num2str(n), '.dat'], ss, 'delimiter', ...
            ' ', 'precision', 8);
    end     

    fatherpath = pwd; 
    output_dir = [fatherpath,'/raw2/lsx', num2str(ls_x), '_lsy', num2str(ls_y),'_var', num2str(VarY),'_smax',num2str(source_max),'_D1r0.1',...
                '/kle',num2str(kle_num), '_', doe, num2str(n_data(i)), '/output/'];

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    Parallel_Computing = 1;
    if Parallel_Computing == 1
        parpool;
        copyexample(n_data(i));
        parfor j = 1 : n_data(i)
            j
            cond = reshape(K(:, j),ngy,ngx);
            forward_model(output_dir,cond,source(j,:),j,j);
        end           
        copyexample(n_data(i),-1);
        delete(gcp('nocreate'));
    else
        for j = 1 : n_data(i)
            cond = reshape(K(:, j),ngy,ngx);
            forward_model(output_dir,cond,source(j,:),1,j);
        end
    end
end    


