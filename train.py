# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

from training import training_loop
# from training import training_loop_simmim as training_loop
# from training import training_loop_woMap as training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    data       = None, # Training dataset (required): <path>
    data_val   = None, # Validation dataset: <path>, default = None. If none, data_val = data
    dataloader = None, # Dataloader, string
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False
    train_mask_path=None,  # 训练集掩码路径
    val_mask_path=None,  # 验证集掩码路径

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    generator  = None, # Path of the generator class
    wdim       = None,
    zdim       = None,
    discriminator = None, # Path of the discriminator class
    loss = None,
    gamma      = None, # Override R1 gamma: <float>
    pr         = None,
    pl         = None, # Train with path length regularization: <bool>, default = True
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>
    truncation = None, # truncation for training: <float>
    style_mix  = None, # style mixing probability for training: <float>
    ema        = None, # Half-life of the exponential moving average (EMA) of generator weights: <int>
    lr         = None, # learning rate
    lrt        = None, # learning rate of transformer: <float>

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    if data_val is None:
        data_val = data
    if dataloader is None:
        dataloader = 'datasets.dataset_512.ImageFolderMaskDataset'

    # 传递自定义掩码路径到数据集参数
    args.training_set_kwargs = dnnlib.EasyDict(
        class_name=dataloader,
        path=data,
        use_labels=True,
        max_size=None,
        xflip=False,
        mask_path=train_mask_path  # 传递训练集掩码路径
    )
    args.val_set_kwargs = dnnlib.EasyDict(
        class_name=dataloader,
        path=data_val,
        use_labels=True,
        max_size=None,
        xflip=False,
        mask_path=val_mask_path  # 传递验证集掩码路径
    )
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)

    try:
        # 初始化训练集并验证参数
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)
        args.training_set_kwargs.resolution = training_set.resolution
        args.training_set_kwargs.use_labels = training_set.has_labels
        args.training_set_kwargs.max_size = len(training_set)
        desc = training_set.name

        # 初始化验证集
        val_set = dnnlib.util.construct_class_by_name(**args.val_set_kwargs)
        args.val_set_kwargs.resolution = val_set.resolution
        args.val_set_kwargs.use_labels = val_set.has_labels
        args.val_set_kwargs.max_size = len(val_set)

        del training_set, val_set  # 释放内存
    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)

    if cond:
        if not args.training_set_kwargs.use_labels or not args.val_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in labels.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False
        args.val_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2),
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'places256': dict(ref_gpus=8,  kimg=50000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'places512': dict(ref_gpus=8,  kimg=50000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'celeba512': dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)
        spec.mbstd = min(spec.mb // gpus, 4)
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res **2) / spec.mb
        spec.ema = spec.mb * 10 / 32

    if generator is None:
        generator = 'networks.mat.Generator'
    else:
        desc += '-' + generator.split('.')[1]
    if discriminator is None:
        discriminator = 'networks.mat.Discriminator'
    if wdim is None:
        wdim = 512
    if zdim is None:
        zdim = 512
    args.G_kwargs = dnnlib.EasyDict(
        class_name=generator,
        z_dim=zdim,
        w_dim=wdim,
        mapping_kwargs=dnnlib.EasyDict(),
        synthesis_kwargs=dnnlib.EasyDict()
    )
    args.D_kwargs = dnnlib.EasyDict(class_name=discriminator)
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map

    if lr is not None:
        assert isinstance(lr, float)
        spec.lrate = lr
        desc += f'-lr{lr:g}'
    if lrt is not None:
        assert isinstance(lrt, float)
        spec.lrt = lrt
        desc += f'-lrt{lrt:g}'

    if lrt is None:
        args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0, 0.99], eps=1e-8)
    else:
        args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, lrt=spec.lrt, betas=[0, 0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0, 0.99], eps=1e-8)

    if loss is None:
        loss = 'losses.loss.TwoStageLoss'
    else:
        desc += '-' + loss.split('.')[-1]
    args.loss_kwargs = dnnlib.EasyDict(class_name=loss, r1_gamma=spec.gamma)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0
        args.loss_kwargs.style_mixing_prob = 0
        args.D_kwargs.architecture = 'orig'

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if pr is not None:
        assert isinstance(pr, float)
        desc += f'-pr{pr:g}'
        args.loss_kwargs.pcp_ratio = pr

    if pl is None:
        pl = True
    assert isinstance(pl, bool)
    if pl is False:
        desc += f'-nopl'
        args.loss_kwargs.pl_weight = 0

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    if truncation is not None:
        assert isinstance(truncation, float)
        desc += '-tc' + str(truncation)
        args.loss_kwargs.truncation_psi = truncation

    if style_mix is not None:
        assert isinstance(style_mix, float)
        desc += '-sm' + str(style_mix)
        args.loss_kwargs.style_mixing_prob = style_mix

    if ema is not None:
        assert isinstance(ema, int)
        desc += '-ema' + str(ema)
        args.ema_kimg = ema

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6
    elif aug == 'noaug':
        pass
    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(
            class_name='training.augment.AugmentPipe',** augpipe_specs[augpipe]
        )

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume]
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume

    if resume != 'noresume':
        args.ada_kimg = 100
        args.ema_rampup = None

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None
        desc += '-fp32'

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # 初始化分布式训练
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # 初始化训练统计
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # 执行训练循环
    training_loop.training_loop(rank=rank, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# 通用选项
@click.option('--outdir', help='结果保存目录', required=True, metavar='DIR')
@click.option('--gpus', help='GPU数量 [默认: 1]', type=int, metavar='INT')
@click.option('--snap', help='快照间隔 [默认: 50]', type=int, metavar='INT')
@click.option('--metrics', help='评估指标列表 [默认: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='随机种子 [默认: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='打印选项后退出', is_flag=True)

# 数据集选项
@click.option('--data', help='训练数据路径', metavar='PATH', required=True)
@click.option('--data_val', help='验证数据路径', metavar='PATH')
@click.option('--dataloader', help='数据加载器类', type=str, metavar='STRING')
@click.option('--cond', help='是否使用条件标签 [默认: false]', type=bool, metavar='BOOL')
@click.option('--subset', help='使用的图像数量', type=int, metavar='INT')
@click.option('--mirror', help='是否启用水平翻转 [默认: false]', type=bool, metavar='BOOL')
@click.option('--train-mask-path', help='训练集自定义掩码目录路径', type=str, metavar='PATH')  # 添加训练集掩码路径选项
@click.option('--val-mask-path', help='验证集自定义掩码目录路径', type=str, metavar='PATH')  # 添加验证集掩码路径选项

# 基础配置
@click.option('--cfg', help='基础配置 [默认: auto]', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', '12', 'paper1024', 'cifar', 'places256', 'places512', 'celeba512']))
@click.option('--generator', help='生成器类路径', type=str, metavar='STRING')
@click.option('--wdim', help='W空间维度', type=int, metavar='INT')
@click.option('--zdim', help='Z空间维度', type=int, metavar='INT')
@click.option('--discriminator', help='判别器类路径', type=str, metavar='STRING')
@click.option('--loss', help='损失函数类路径', type=str, metavar='STRING')
@click.option('--gamma', help='R1正则化系数', type=float)
@click.option('--pr', help='感知损失权重', type=float)
@click.option('--pl', help='是否启用路径长度正则化 [默认: true]', type=bool, metavar='BOOL')
@click.option('--kimg', help='训练总图像数(千张)', type=int, metavar='INT')
@click.option('--batch', help='批量大小', type=int, metavar='INT')
@click.option('--truncation', help='截断系数', type=float)
@click.option('--style-mix', help='风格混合概率', type=float)
@click.option('--ema', help='EMA半衰期', type=int, metavar='INT')
@click.option('--lr', help='学习率', type=float)
@click.option('--lrt', help='Transformer学习率', type=float)

# 判别器增强
@click.option('--aug', help='增强模式 [默认: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--p', help='固定增强概率', type=float)
@click.option('--target', help='ADA目标值', type=float)
@click.option('--augpipe', help='增强管道 [默认: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']))

# 迁移学习
@click.option('--resume', help='恢复训练的模型路径', metavar='PKL')
@click.option('--freezed', help='冻结判别器层数', type=int, metavar='INT')

# 性能选项
@click.option('--fp32', help='禁用混合精度', type=bool, metavar='BOOL')
@click.option('--nhwc', help='使用NHWC内存格式', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='允许TF32计算', type=bool, metavar='BOOL')
@click.option('--nobench', help='禁用cuDNN基准测试', type=bool, metavar='BOOL')
@click.option('--workers', help='数据加载进程数', type=int, metavar='INT')

def main(ctx, outdir, dry_run,** config_kwargs):
    dnnlib.util.Logger(should_flush=True)

    # 解析训练参数
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # 确定输出目录
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # 打印配置信息
    print('\n训练配置:')
    print(json.dumps(args, indent=2))
    print(f'\n输出目录: {args.run_dir}')
    print(f'训练数据: {args.training_set_kwargs.path}')
    print(f'自定义掩码路径: {args.training_set_kwargs.mask_path if args.training_set_kwargs.mask_path else "未指定(使用随机掩码)"}')
    print(f'训练总图像数: {args.total_kimg} kimg')
    print(f'GPU数量: {args.num_gpus}')
    print(f'图像分辨率: {args.training_set_kwargs.resolution}')
    print()

    #  dry-run模式
    if dry_run:
        print('dry-run模式，退出。')
        return

    # 创建输出目录
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # 启动训练进程
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter