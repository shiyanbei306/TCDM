# TCDM

### Download models
Here are the download links for each model checkpoint:
* CIFAR100LT 32x32: [TCDM.pt](https://drive.google.com/drive/folders/1Gt8prL4cloPbQrY6XHKhZgxOxKVcwAds?dmr=1&ec=wgc-drive-globalnav-goto)

### Sampling 
  
 ``` python main.py --flagfile ./logs/cifar100lt_cbdm_c100/flagfile.txt --logdir ./logs/cifar100lt_cbdm_c100 --fid_cache ./stats/cifar10.train.npz  --ckpt_step 300000 --num_images 50000 --batch_size 64 --notrain --eval --sample_method cfg  --omega 1.4 --sample_type DDPM ```

### Calculating FID
Please refer to [FID](https://github.com/forever208/ADM-ES/blob/ADM-ES/evaluations/README.md) for the full instructions

### Acknowledgements

This implementation is based on / inspired by:

- [https://github.com/w86763777/pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) 
- [https://github.com/qym7/CBDM-pytorch](https://github.com/qym7/CBDM-pytorch.git)
