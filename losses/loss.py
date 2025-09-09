# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from losses.pcp import PerceptualLoss
import torch.nn.functional as F


# ----------------------------------------------------------------------------

class Loss:
    # 这是一个抽象基类，定义了所有损失函数都必须实现的 accumulate_gradients 方法。
    def accumulate_gradients(self, phase, real_img, mask, real_c, gen_z, gen_c, sync, gain):  # 子类需要重写此方法
        raise NotImplementedError()


# ----------------------------------------------------------------------------

def calculate_psnr(img1, img2):
    """计算两张图像之间的峰值信噪比(PSNR)。"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class TwoStageLoss(Loss):
    """
    用于两阶段图像修复模型的损失函数。
    包含了对抗性损失、感知损失、L1损失，以及针对遥感图像高光修复定制的光谱角匹配损失和高光抑制损失。
    """

    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, truncation_psi=1, pcp_ratio=1.0, sam_weight=1.0,
                 highlight_weight=0.5):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping  # 生成器的映射网络
        self.G_synthesis = G_synthesis  # 生成器的合成网络
        self.D = D  # 判别器
        self.augment_pipe = augment_pipe  # 数据增强管道
        self.style_mixing_prob = style_mixing_prob  # 风格混合概率
        self.r1_gamma = r1_gamma  # R1正则化系数
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.truncation_psi = truncation_psi
        # 初始化感知损失
        self.pcp = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(device)
        self.pcp_ratio = pcp_ratio  # 感知损失的权重
        self.sam_weight = sam_weight  # 光谱角匹配损失的权重
        self.highlight_weight = highlight_weight  # 高光抑制损失的权重

    def run_G(self, img_in, mask_in, z, c, sync):
        """运行生成器，生成修复后的图像。"""
        # 运行映射网络，从Z空间映射到W空间
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c, truncation_psi=self.truncation_psi)
            # 执行风格混合
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, truncation_psi=self.truncation_psi,
                                                    skip_w_avg_update=True)[:, cutoff:]
        # 运行合成网络，生成图像
        with misc.ddp_sync(self.G_synthesis, sync):
            img, img_stg1 = self.G_synthesis(img_in, mask_in, ws, return_stg1=True)
        return img, ws, img_stg1

    def run_D(self, img, mask, img_stg1, c, sync):
        """运行判别器，获取判别分数。"""
        with misc.ddp_sync(self.D, sync):
            logits, logits_stg1 = self.D(img, mask, img_stg1, c)
        return logits, logits_stg1

    def spectral_angle_mapper_loss(self, gen_img, real_img, mask):
        """
        计算光谱角匹配损失 (SAM Loss)。
        此损失仅在图像的已知区域（未被掩码遮挡的区域）上计算，以确保修复区域的光谱特征与周围环境一致。
        """
        # 掩码中，已知区域为1，未知（待修复）区域为0。
        valid_mask = mask

        # 关键修改 1: 将图像的数值范围从 [-1, 1] 变换到 [0, 1]，以便进行稳定且有物理意义的计算。
        gen_img_norm = (gen_img + 1.0) / 2.0
        real_img_norm = (real_img + 1.0) / 2.0

        # 将掩码应用于生成图像和真实图像，使计算聚焦于已知区域
        gen_known = gen_img_norm * valid_mask
        real_known = real_img_norm * valid_mask

        # 关键修改 2: 增加一个极小值 epsilon 以保证数值稳定性，避免除以零。
        epsilon = 1e-8

        # 对每个像素的光谱向量进行归一化
        gen_norm = F.normalize(gen_known, p=2, dim=1, eps=epsilon)
        real_norm = F.normalize(real_known, p=2, dim=1, eps=epsilon)

        # 计算余弦相似度
        cos_similarity = (gen_norm * real_norm).sum(dim=1)

        # 通过反余弦计算角度。使用 clamp 函数防止因浮点数精度问题导致值略微超出[-1, 1]范围而产生NaN。
        sam_loss = torch.acos(cos_similarity.clamp(-1, 1)).mean()

        return sam_loss

    def highlight_suppression_loss(self, gen_img, mask):
        """
        计算高光抑制损失。
        此损失用于惩罚在修复区域（掩码区域）内生成新的高亮度像素点。
        """
        # 掩码中，待修复区域为0。我们希望在此区域抑制高光，所以反转掩码。
        inpainted_region_mask = 1 - mask

        # 定义高光的亮度阈值（例如，在[-1, 1]范围内大于0.8，约等于[0, 255]范围内的230）
        highlight_threshold = 0.8

        # 计算生成图像的平均亮度
        gen_intensity = torch.mean(gen_img.float(), dim=1, keepdim=True)

        # 找到那些既在修复区域内、亮度又高于阈值的像素
        new_highlights = (gen_intensity > highlight_threshold).float() * inpainted_region_mask

        # 损失值即为这些新高光像素的平均亮度，以此作为惩罚
        highlight_loss = (gen_intensity * new_highlights).sum() / (new_highlights.sum() + 1e-8)

        return highlight_loss

    def accumulate_gradients(self, phase, real_img, mask, real_c, gen_z, gen_c, sync, gain):
        """计算并累积梯度。这是训练的核心步骤。"""
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])  # 是否执行生成器主训练
        do_Dmain = (phase in ['Dmain', 'Dboth'])  # 是否执行判别器主训练
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)  # 是否执行生成器路径长度正则化
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)  # 是否执行判别器R1正则化

        # --- 生成器主训练 ---
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # 1. 通过生成器获取生成图像
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=(sync and not do_Gpl))
                # 2. 将生成图像送入判别器，获取判别分数
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False)

                # 记录对抗性损失
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # 目标是最大化 gen_logits
                training_stats.report('Loss/G/loss', loss_Gmain)

                loss_Gmain_stg1 = torch.nn.functional.softplus(-gen_logits_stg1)
                training_stats.report('Loss/G/loss_s1', loss_Gmain_stg1)

                # 3. 计算重建损失 (L1, 感知损失, PSNR)
                l1_loss = torch.mean(torch.abs(gen_img - real_img))
                training_stats.report('Loss/G/L1', l1_loss)
                pcp_loss, _ = self.pcp(gen_img, real_img)
                training_stats.report('Loss/G/PCP', pcp_loss)

                # PSNR仅用于监控，不直接参与梯度计算
                gen_img_psnr = (gen_img.clamp(-1, 1) + 1) * 127.5
                real_img_psnr = (real_img + 1) * 127.5
                psnr = calculate_psnr(gen_img_psnr, real_img_psnr)
                training_stats.report('Loss/G/PSNR', psnr)

                # 4. 计算自定义的遥感图像损失
                sam_loss = self.spectral_angle_mapper_loss(gen_img, real_img, mask)
                training_stats.report('Loss/G/SAM', sam_loss)
                highlight_loss = self.highlight_suppression_loss(gen_img, mask)
                training_stats.report('Loss/G/Highlight', highlight_loss)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                # 5. 将所有生成器损失加权求和，并进行反向传播
                loss_Gmain_all = (loss_Gmain + loss_Gmain_stg1 +
                                  pcp_loss * self.pcp_ratio +
                                  sam_loss * self.sam_weight +
                                  highlight_loss * self.highlight_weight)
                loss_Gmain_all.mean().mul(gain).backward()

        # --- 判别器主训练 ---
        if do_Dmain:
            # 1. 对生成图像的判别
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=False)
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # 目标是最小化 gen_logits

                loss_Dgen_stg1 = torch.nn.functional.softplus(gen_logits_stg1)
                training_stats.report('Loss/scores/fake_s1', gen_logits_stg1)
                training_stats.report('Loss/signs/fake_s1', gen_logits_stg1.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen_all = loss_Dgen + loss_Dgen_stg1
                loss_Dgen_all.mean().mul(gain).backward()

        # 2. 对真实图像的判别，并计算R1正则化
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                mask_tmp = mask.detach().requires_grad_(do_Dr1)
                real_img_tmp_stg1 = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_logits_stg1 = self.run_D(real_img_tmp, mask_tmp, real_img_tmp_stg1, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_Dreal_stg1 = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # 目标是最大化 real_logits
                    loss_Dreal_stg1 = torch.nn.functional.softplus(-real_logits_stg1)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/loss_s1', loss_Dgen_stg1 + loss_Dreal_stg1)

                loss_Dr1 = 0
                loss_Dr1_stg1 = 0
                if do_Dr1:  # R1 正则化，惩罚判别器对真实图像的梯度，以稳定训练
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                        torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                            only_inputs=True)[0]
                        r1_grads_stg1 = \
                        torch.autograd.grad(outputs=[real_logits_stg1.sum()], inputs=[real_img_tmp_stg1],
                                            create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                    r1_penalty_stg1 = r1_grads_stg1.square().sum([1, 2, 3])
                    loss_Dr1_stg1 = r1_penalty_stg1 * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_s1', r1_penalty_stg1)
                    training_stats.report('Loss/D/reg_s1', loss_Dr1_stg1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                # 将真实图像损失和正则化损失合并进行反向传播
                (loss_Dreal + loss_Dreal_stg1 + loss_Dr1 + loss_Dr1_stg1).mean().mul(gain).backward()

