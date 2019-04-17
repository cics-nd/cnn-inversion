clear all,close all

Npar = 686;
Par.Nkle = 679;
Par.MeanY = 2.0;
Nkle = 23;

Ne = 6000;

load results.mat % load results of surrogate-based ILUES
x = xall(Par.Nkle+1:Npar,:);

% load results of original ILUES
data = load(['/afs/crc.nd.edu/user/s/smo/invers_mt3d/ILUES_168obs/','x',num2str(1),'.mat']);
xall_ILUES = data.x1;
for i = 1 : 20
    data = load(['/afs/crc.nd.edu/user/s/smo/invers_mt3d/ILUES_168obs/','xa',num2str(i),'.mat']);
    xall_ILUES = [xall_ILUES,data.xa];
end
x_ILUES = xall_ILUES(Par.Nkle+1:Npar,:);

Niter = floor(size(xall,2)/Ne);

y_label = {'\itS_{lx}','\itS_{ly}','\itS_{s\rm1}','\itS_{s\rm2}','\itS_{s\rm3}','\itS_{s\rm4}','\itS_{s\rm5}'};

xt = load('obs/ss347.dat');

figure,
% f = figure('visible','off');

for i = 1 : size(x_ILUES,1)
    xi = zeros(Ne,Niter);
    for j = 1 : Niter
        n1 = 1+(j-1)*Ne;
        n2 = j*Ne;
        xi(:,j) = (x_ILUES(i,n1:n2))';
    end
    subplot(4,4,i),

    boxplot(xi,'Colors','k','Symbol','.b','OutlierSize',2,'Widths',0.6)%,'Whisker',1)
    pbaspect([2 1.3 1])
    lines = findobj(gcf,'Tag','Median');
    set(lines,'Color','k');
    xticks([1 5 9 13 17 21])
    xticklabels({'0','4','8','12','16','20'})
    set(gca,'XMinorTick','on','YMinorTick','on')
    hold on,
    plot([1,21],[xt(i),xt(i)],'--r','LineWidth',1.2),
    ylabel(y_label{i});
    xlabel('Number of iterations'),
    hold on,
end



for i = 1 : size(x,1)
    xi = zeros(Ne,Niter);
    for j = 1 : Niter
        n1 = 1+(j-1)*Ne;
        n2 = j*Ne;
        xi(:,j) = (x(i,n1:n2))';
    end
    subplot(4,4,i+8),

    boxplot(xi,'Colors','k','Symbol','.b','OutlierSize',2,'Widths',0.6)%,'Whisker',1)
    pbaspect([2 1.3 1])
    lines = findobj(gcf,'Tag','Median');
    set(lines,'Color','k');
    xticks([1 5 9 13 17 21])
    xticklabels({'0','4','8','12','16','20'})
    set(gca,'XMinorTick','on','YMinorTick','on')
    hold on,
    plot([1,21],[xt(i),xt(i)],'--r','LineWidth',1.2),
    ylabel(y_label{i});
    xlabel('Number of iterations'),
    hold on,
end
hold off
% saveas(f,'boxplot','fig')
