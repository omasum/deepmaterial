conda activate ghpy37
cd /mnt/hard_disk/gaihe/code/DeepMateral
python deepmateral/train.py -opt options/train/MSVR/train_MSVR_Pcd8TSAReconsMeta_L1_F7G064Feat64.yml

python deepmateral/ms_test.py -opt options/test/MSVR/test_MSVR_EDVRM.yml

python deepmateral/train.py -opt options/train/META/train_META_Vimeo90K_F64B16BN4.yml
# bicubic
# scale: [2.0, 2.1,.....,4.0]
# psnr: 35.6883, 35.2793, 34.8285, 34.4180, 34.0369, 33.6744, 33.3318, 33.0142, 32.7154, 32.4254, 32.1582, 31.8876, 31.6527, 31.4035, 31.1843, 30.9713, 30.7582, 30.5609, 30.3762, 30.1929, 30.0345
# ssim: 0.9510, 0.9510, 0.9467, 0.9416, 0.9364, 0.9312, 0.9259, 0.9207, 0.9155, 0.9104, 0.9052, 0.9003, 0.8951, 0.8902, 0.8851, 0.8802, 0.8755, 0.8706, 0.8659, 0.8616, 0.8570, 0.8528
