@echo off
:: 设置 VS 构建环境
call "D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: 激活 Conda 环境
call D:\anaconda3\Scripts\activate.bat
call conda activate MAT

:: 切换工作目录
cd /d D:\MAT2

:: 启动训练脚本
python train.py ^
    --outdir="D:\MAT2\shuchu" ^
    --gpus=1 ^
    --batch=1 ^
    --metrics=none ^
    --data="D:\xunlianshuju\zongfenge\train" ^
    --data_val="D:\xunlianshuju\zongfenge\val" ^
    --dataloader=datasets.dataset_512.ImageFolderMaskDataset ^
    --mirror=True ^
    --cond=False ^
    --cfg=places512 ^
    --aug=noaug ^
    --generator=networks.mat.Generator ^
    --discriminator=networks.mat.Discriminator ^
    --loss=losses.loss.TwoStageLoss ^
    --pr=0.1 ^
    --pl=False ^
    --truncation=0.5 ^
    --style-mix=0.5 ^
    --ema=1 ^
    --lr=0.001 ^
    --kimg=20 ^
    --workers=1 ^
    --snap=5 ^
    --resume="D:\MAT\shuchu\00036-train-mirror-places512-mat-lr0.001-TwoStageLoss-pr0.1-nopl-kimg10-batch1-tc0.5-sm0.5-ema1-noaug-resumecustom\network-snapshot-000010.pkl"
:: 防止窗口一闪而过
pause