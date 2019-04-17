function []= copyexample(NumPoints,flag)
current_path = pwd;
cd([current_path,'/example']);

if nargin < 2  
    parfor i=2:NumPoints
        copyfile('model_1',['model_',num2str(i)]);
    end    
elseif flag == -1
    parfor i = 2:NumPoints
        rmdir(['model_',num2str(i)],'s');
    end
end

cd(current_path);