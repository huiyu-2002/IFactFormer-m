# Implicit factorized transformer approach to fast prediction of turbulent channel flows

Code and data accompanying the manuscript titled ["Implicit factorized transformer approach to fast prediction of turbulent channel flows"](https://arxiv.org/abs/2412.18840 ), authored by Huiyu Yang, Yunpeng Wang and Jianchun Wang.



## Abstract

Transformer neural operators have recently become an effective approach for surrogate modeling of systems governed by partial differential equations (PDEs). In this paper, we introduce a modified implicit factorized transformer (IFactFormer-m) model which replaces the original chained factorized attention with parallel factorized attention. The IFactFormer-m model successfully performs long-term predictions for turbulent channel flow, whereas the original IFactFormer (IFactFormer-o), Fourier neural operator (FNO), and implicit Fourier neural operator (IFNO) exhibit a poor performance. Turbulent channel flows are simulated by direct numerical simulation using fine grids at friction Reynolds numbers $\text{Re}_{\tau}\approx 180,395,590$, and filtered to coarse grids for training neural operator. The neural operator takes the current flow field as input and predicts the flow field at the next time step, and long-term prediction is achieved in the posterior through an autoregressive approach. The results show that IFactFormer-m, compared to other neural operators and the traditional large eddy simulation (LES) methods including dynamic Smagorinsky model (DSM) and the wall-adapted local eddy-viscosity (WALE) model, reduces prediction errors in the short term, and achieves stable and accurate long-term prediction of various statistical properties and flow structures, including the energy spectrum, mean streamwise velocity, root mean square (rms) values of fluctuating velocities, Reynolds shear stress, and spatial structures of instantaneous velocity. Moreover, the trained IFactFormer-m is much faster than traditional LES methods. By analyzing the attention kernels, we elucidate the reasons why IFactFormer-m converges faster and achieves a stable and accurate long-term prediction compared to IFactFormer-o. 



## Dataset

The dataset can download at [fDNS_kaggle](https://www.kaggle.com/datasets/aifluid/coarsened-fdns-data-chl).



## Citation

```
@article{yang2026implicit,
  title={Implicit factorized transformer approach to fast prediction of turbulent channel flows},
  author={Yang, Huiyu and Wang, Yunpeng and Wang, Jianchun},
  journal={Science China Physics, Mechanics & Astronomy},
  volume={69},
  number={1},
  pages={214606},
  year={2026}
}
```