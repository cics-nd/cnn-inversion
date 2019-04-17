function x1 = RangeTrans(x,range)

xlow = repmat(range(1,:),size(x,1),1);
xup =  repmat(range(2,:),size(x,1),1);
% x1 = (x - xlow) ./ (xup - xlow);
x1 = x .* (xup - xlow) + xlow;
end