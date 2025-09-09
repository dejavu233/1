# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

import legacy
from metrics import metric_main
from metrics import metric_utils


# ----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    """设置快照图像网格，用于可视化训练过程中的生成样本。"""
    rnd = np.random.RandomState(random_seed)
    # 计算网格的宽度和高度
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # 如果数据集没有标签，则随机显示一部分训练样本
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # 如果有标签，则按标签对训练样本进行分组
        label_groups = dict()  # 字典：标签 => [索引, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # 重新排序
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # 将样本组织到网格中
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # 加载数据
    images, masks, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(masks), np.stack(labels)


# ----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    """将图像网格保存为图片文件。"""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    # 将数值范围从drange映射到[0, 255]
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    # 重新排列数组维度以匹配图像格式
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:  # 灰度图
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:  # 彩色图
        PIL.Image.fromarray(img, 'RGB').save(fname)


# ----------------------------------------------------------------------------

def training_loop(
        run_dir='.',  # 输出目录
        training_set_kwargs={},  # 训练集选项
        val_set_kwargs={},  # 验证集选项
        data_loader_kwargs={},  # 数据加载器选项
        G_kwargs={},  # 生成器网络选项
        D_kwargs={},  # 判别器网络选项
        G_opt_kwargs={},  # 生成器优化器选项
        D_opt_kwargs={},  # 判别器优化器选项
        augment_kwargs=None,  # 数据增强选项
        loss_kwargs={},  # 损失函数选项
        metrics=[],  # 训练期间评估的指标
        random_seed=0,  # 全局随机种子
        num_gpus=1,  # 参与训练的GPU数量
        rank=0,  # 当前进程的排名 [0, num_gpus-1]
        batch_size=4,  # 一次训练迭代的总批量大小
        batch_gpu=4,  # 每个GPU一次处理的样本数
        ema_kimg=10,  # 生成器权重的指数移动平均(EMA)的半衰期
        ema_rampup=None,  # EMA预热系数
        G_reg_interval=4,  # G的正则化频率
        D_reg_interval=16,  # D的正则化频率
        augment_p=0,  # 数据增强概率的初始值
        ada_target=None,  # ADA目标值
        ada_interval=4,  # ADA调整频率
        ada_kimg=500,  # ADA调整速度
        total_kimg=25000,  # 训练总长度（千张图片）
        kimg_per_tick=1,  # 进度快照间隔
        image_snapshot_ticks=50,  # 保存图像快照的频率
        network_snapshot_ticks=50,  # 保存网络快照的频率
        resume_pkl=None,  # 用于恢复训练的网络模型文件
        cudnn_benchmark=True,  # 是否启用cuDNN基准测试以提高速度
        allow_tf32=False,  # 是否允许PyTorch内部使用TF32进行计算
        abort_fn=None,  # 用于确定是否中止训练的回调函数
        progress_fn=None,  # 用于更新训练进度的回调函数
        **kwargs
):
    # 1. 初始化
    # -------------------------------------------------------------------
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # 2. 加载数据集
    # -------------------------------------------------------------------
    if rank == 0:
        print('正在加载训练集...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    val_set = dnnlib.util.construct_class_by_name(**val_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print(f'图片数量: {len(training_set)}')
        print(f'图片形状: {training_set.image_shape}')
        print(f'标签形状: {training_set.label_shape}')
        print()

    # 3. 构建网络
    # -------------------------------------------------------------------
    if rank == 0:
        print('正在构建网络...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution,
                         img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    # 4. 从现有模型恢复训练
    # -------------------------------------------------------------------
    if (resume_pkl is not None) and (rank == 0):
        print(f'正在从 "{resume_pkl}" 恢复...')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # 5. 打印网络结构摘要
    # -------------------------------------------------------------------
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img_in = torch.empty([batch_gpu, training_set.num_channels, training_set.resolution, training_set.resolution],
                             device=device)
        mask_in = torch.empty([batch_gpu, 1, training_set.resolution, training_set.resolution], device=device)
        img = misc.print_module_summary(G, [img_in, mask_in, z, c])
        img_stg1 = torch.empty([batch_gpu, 3, training_set.resolution, training_set.resolution], device=device)
        misc.print_module_summary(D, [img, mask_in, img_stg1, c])

    # 6. 设置数据增强
    # -------------------------------------------------------------------
    if rank == 0:
        print('正在设置数据增强...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # 7. 分布式训练设置
    # -------------------------------------------------------------------
    if rank == 0:
        print(f'正在分发到 {num_gpus} 个GPU...')
    ddp_modules = dict()
    for name, module in [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema),
                         ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # 8. 设置训练阶段、损失函数和优化器
    # -------------------------------------------------------------------
    if rank == 0:
        print('正在设置训练阶段...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval),
                                                   ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # 惰性正则化
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            if 'lrt' in opt_kwargs:
                filter_list = ['tran', 'Tran']
                base_params = []
                tran_params = []
                for pname, param in module.named_parameters():
                    flag = False
                    for fname in filter_list:
                        if fname in pname:
                            flag = True
                    if flag:
                        tran_params.append(param)
                    else:
                        base_params.append(param)
                optim_params = [{'params': base_params}, {'params': tran_params, 'lr': opt_kwargs.lrt * mb_ratio}]
                optim_kwargs = dnnlib.EasyDict()
                for key, val in opt_kwargs.items():
                    if 'lrt' != key:
                        optim_kwargs[key] = val
            else:
                optim_params = module.parameters()
                optim_kwargs = opt_kwargs
            opt = dnnlib.util.construct_class_by_name(optim_params, **optim_kwargs)
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # 9. 导出样本图像
    # -------------------------------------------------------------------
    grid_size = None
    grid_z = None
    grid_c = None
    grid_img = None
    grid_mask = None
    if rank == 0:
        print('正在导出样本图像...')
        grid_size, images, masks, labels = setup_snapshot_image_grid(training_set=val_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        save_image_grid(masks, os.path.join(run_dir, 'masks.png'), drange=[0, 1], grid_size=grid_size)

        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_img = (torch.from_numpy(images).to(device) / 127.5 - 1).split(batch_gpu)
        grid_mask = torch.from_numpy(masks).to(device).split(batch_gpu)
        images = torch.cat([G_ema(img_in, mask_in, z, c, noise_mode='const').cpu() for img_in, mask_in, z, c in
                            zip(grid_img, grid_mask, grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size)

    # 10. 初始化日志
    # -------------------------------------------------------------------
    if rank == 0:
        print('正在初始化日志...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('跳过TensorBoard日志导出:', err)

    # 11. 核心训练循环
    # -------------------------------------------------------------------
    if rank == 0:
        print(f'开始训练，总计 {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    total_images = total_kimg * 1000
    if rank == 0:
        pbar = tqdm(total=total_images, desc="训练进度", unit="images")

    while True:
        # 获取一个批次的数据
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_mask, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_mask = phase_mask.to(device).to(torch.float32).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                         range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # 执行训练阶段
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for round_idx, (real_img, mask, real_c, gen_z, gen_c) in enumerate(
                    zip(phase_real_img, phase_mask, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, mask=mask, real_c=real_c, gen_z=gen_z,
                                          gen_c=gen_c, sync=sync, gain=gain)
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # 更新G_ema
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # 更新状态
        cur_nimg += batch_size
        batch_idx += 1
        if rank == 0:
            pbar.update(batch_size)

        # 执行ADA
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (
                        ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # 每个tick执行一次维护任务
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # 12. 打印状态、保存快照、评估指标
        # -------------------------------------------------------------------

        # ** 修正开始 **
        # 1. 在打印前先收集和处理本轮的统计数据
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # 2. 打印状态行
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))

        if rank == 0:
            # 打印第一行：主要的训练进度信息
            print(' '.join(fields))

            # 3. 从处理好的 stats_dict 中安全地获取损失值并打印第二行
            loss_fields = []
            # 使用 .get(key, default_value).mean 的方式确保即使某个值不存在也不会报错
            loss_fields += [f"L1 {stats_dict.get('Loss/G/L1', dnnlib.EasyDict(mean=float('nan'))).mean:.4f}"]
            loss_fields += [f"PCP {stats_dict.get('Loss/G/PCP', dnnlib.EasyDict(mean=float('nan'))).mean:.4f}"]
            loss_fields += [f"PSNR {stats_dict.get('Loss/G/PSNR', dnnlib.EasyDict(mean=float('nan'))).mean:.2f}"]
            loss_fields += [f"SAM {stats_dict.get('Loss/G/SAM', dnnlib.EasyDict(mean=float('nan'))).mean:.4f}"]
            loss_fields += [
                f"Highlight {stats_dict.get('Loss/G/Highlight', dnnlib.EasyDict(mean=float('nan'))).mean:.4f}"]
            print(f"    {' '.join(loss_fields)}")
        # ** 修正结束 **

        # 检查是否中止
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('中止训练...')

        # 保存图像快照
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(img_in, mask_in, z, c, noise_mode='const').cpu() for img_in, mask_in, z, c in
                                zip(grid_img, grid_mask, grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size)

        # 保存网络快照
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs), val_set_kwargs=dict(val_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=[r'.*\.w_avg', r'.*\.relative_position_index',
                                                                         r'.*\.avg_weight', r'.*\.attn_mask',
                                                                         r'.*\.resample_filter'])
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # 评估指标
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                torch.cuda.empty_cache()
                print('正在评估指标...')
            for metric in metrics:
                valid_kwargs = {k: v for k, v in kwargs.items() if
                                k in metric_utils.MetricOptions.__init__.__code__.co_varnames}
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                                      dataset_kwargs=val_set_kwargs, num_gpus=num_gpus, rank=rank,
                                                      device=device, **valid_kwargs)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data

        # 更新日志
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # 更新状态
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # 训练结束
    if rank == 0:
        pbar.close()
        print()
        print('退出...')

