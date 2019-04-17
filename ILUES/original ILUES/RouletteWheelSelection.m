function index=RouletteWheelSelection(V,m)
% Input:
%      V           -----fitness criterion
%      m           -----number of individuals to be chosen
% Output:
%      index       -----index of the chosen individuals


n=size(V,2);
    if max(V)==0&&min(V)==0
        index=ceil(rand(1,m)*n);
    else
        temindex=find(V~=0);
        n=length(temindex);
        V=V(temindex);

        V=cumsum(V)/sum(V);

        pp=rand(1,m);
        k = 1;
        for i=1:m,

            while 1
                flag = 1;
                for j=1:n,
                    if pp(i) < V(j)
                        index(k) = j;
                        k = k+ 1;
                        V(j) = 0;
                        flag = 0;
                        break
                    end
                end
                if flag 
                    pp(i) = rand;
                else
                    break;
                end
            end 
        end
    end