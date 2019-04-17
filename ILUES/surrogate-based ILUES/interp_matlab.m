function [] = interp_matlab()

% observation locations
xobs = [5.7,10.0,14.8,5.8,12.5,17.5, 6.1, 10.0, 18.2, 12.8 10.8, 12.2, 7.2, 8.8, 18.0, 6.8, 8.0, 8.4, 14.5,16.0,10.5]; 
yobs = [3.8,2.4, 3.6, 4.9, 5.0, 3.4, 6.3,  7.1,  4.5,  2.9, 3.9,  6.9, 5.5, 5.9,  6.1, 4.3, 2.2, 4.0, 5.5, 7.0, 5.5];

load InterpData.mat
y_sim = nan(1, 168);
for i = 1 : size(y_pred,1)
	y_sim(1, 1+(i-1)*21:i*21) = interp2(x,y,squeeze( y_pred(i,:,:) ),xobs,yobs,'linear');
end
save y_sim.dat y_sim -ascii 

end
