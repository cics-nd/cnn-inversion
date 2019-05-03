## Deep autoregressive neural networks for high-dimensional inverse problems

[Deep autoregressive neural networks for high-dimensional inverse problems in groundwater contaminant source identification](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR024638)

[Shaoxing Mo](https://scholar.google.com/citations?hl=en&user=G6ac1xUAAAAJ&view_op=list_works&gmla=AJsN-F4ses_YhFsF-w2sFZLhacR7vrVyN1272g_B7XQyGbYsvy_6ReJpe4ChndNy_cFQ7UqXCSi82UiLjMB2dKyqSj8x5DaPRg), [Nicholas Zabaras](https://www.zabaras.com/), [Xiaoqing Shi](https://scholar.google.com/citations?user=MLKqgKoAAAAJ&hl=en&oi=sra), Jichun Wu

PyTorch implementation of deep autoregressive nueral networks based on a dense convolutional encoder-decoder network architecture for dynamical solute transport models with a time-varying source term and for subsequent high-dimensional inverse modeling. In the network, the time-varying process is represented using an autoregressive model, in which the time-dependent output at previous time step (*y*<sub>*i*-1</sub>) is treated as input to predict the current output (*y*<sub>*i*</sub>), that is, 

**_y_<sub>_i_</sub>=_f_(_x_<sub>_i_</sub>,_y_<sub>_i_-1</sub>)**, 

where x is the uncertain model input considered.

# Dependencies
* python 3
* PyTorch 0.4
* h5py
* matplotlib
* seaborn

# Datasets, Pretrained Model, and Forward Model Input Files
The datasets used have been uploaded to Google Drive and can be downloaded using this link [https://drive.google.com/drive/folders/1CnITMyMOTmuSHQp8p5G9Vju3SFzi-9ae?usp=sharing](https://drive.google.com/drive/folders/1CnITMyMOTmuSHQp8p5G9Vju3SFzi-9ae?usp=sharing)

# Training Data Shape
The training data are saved in the form: N x Nc x H x W, where N is the number of training samples, Nc is the number of input/output channels (i.e., the number of input/output fields considered), H x W is the spatial discretization resolution of the domain.

# Network Training
With the training data prepared with the shape mentioned above, use the following command to train the network:
```
python train_Net.py   OR   python3 train_Net.py
```
One will need to change the 'data-dir' parameter, probably need to modify the values of kernel size, stride, zero padding in dense_ed.py according to the value of H x W (see Section 4.4 in [Mo et al. (2019)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR024638) for details).

# Inverse Modeling
The iterative local updating ensemble smoother (ILUES) algorithm proposed in [Zhang et al. (2018)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017WR020906) is used in this study as the inversion framework to solve high-dimensional inverse problems. We would like to thank Dr. Zhang for sharing the codes of ILUES.

# Citation
See [Mo et al. (2019)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR024638) for more information. If you find this repo useful for your research, please consider to cite:
```
@article{moetal2019,
author = {Mo, Shaoxing and Zabaras, Nicholas and Shi, Xiaoqing and Wu, Jichun},
title = {Deep autoregressive neural networks for high-dimensional inverse problems in groundwater contaminant
         source identification},
journal = {Water Resources Research},
volume = {},
number = {},
pages = {},
year = {2019}
doi = {10.1029/2018WR024638},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2018WR024638}
}
```
or:
```
Mo, S., Zabaras, N., Shi, X., & Wu, J. (2019). Deep autoregressive neural networks for high‐dimensional inverse
problems in groundwater contaminant source identification. Water Resources Research, 55. 
https://doi.org/10.1029/2018WR024638
```

# Questions
Contact Shaoxing Mo (smo@smail.nju.edu.cn) or Nicholas Zabaras (nzabaras@gmail.com) with questions or comments.
