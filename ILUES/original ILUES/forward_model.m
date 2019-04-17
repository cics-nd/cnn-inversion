function [obs] = forward_model(cond,source,i)

filepath = pwd;

cd([filepath,'/example/model_',num2str(i)]);  

obs = forwardsys280obsnew(cond,source);

cd(filepath);

end

