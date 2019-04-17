function []=forward_model(output_dir,cond,source,i,j)
filepath=pwd;
cd([filepath,'/example/model_',num2str(i)]);
forwardsys280obsnew(output_dir,cond,source,j);
cd(filepath);

end