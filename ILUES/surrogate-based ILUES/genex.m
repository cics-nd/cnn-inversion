function x = genex(range,N,Npar,Nkle)

if nargin<2
    N=1;
end

x = nan(Npar,N);
for i=1:N
    x(1:Nkle,i) = randn(Nkle,1);
    x(Nkle+1:Npar,i) = range(Nkle+1:Npar,1) + (range(Nkle+1:Npar,2) - range(Nkle+1:Npar,1)).*rand(Npar-Nkle,1);
end

% Boundary handling
for i = 1:N
    for j = 1:Nkle
        if x(j,i) > range(j,2)
            x(j,i) =  2*range(j,2) - x(j,i); % (range(j,2) + x(j,i))/2;
        elseif x(j,i) < range(j,1)
            x(j,i) = 2*range(j,1) - x(j,i); %(range(j,1) + x(j,i))/2;
        end
    end
end

end