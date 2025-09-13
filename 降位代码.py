import cv2
import numpy as np
import os
from tqdm import tqdm

# =================================================================================
# === 1. 在这里配置文件夹路径 ===

# --- 输入路径 ---
# 包含高位深(如16位)遥感图像的文件夹 (通常是 .tif 格式)
INPUT_FOLDER_16BIT = r"high_bit_depth_images_16bit"

# --- 输出路径 ---
# 保存降位后的8位图像的文件夹
OUTPUT_FOLDER_8BIT = r"converted_images_8bit"

# =================================================================================
# === 2. 在这里配置降位算法参数 ===

# --- 降位方法选择 ---
# 可选: 'changguang_transform', 'percentile_stretch'
# changguang_transform: 模拟您截图中的“长光变换”算法
# percentile_stretch:   标准的百分比截断线性拉伸
STRETCH_METHOD = 'changguang_transform'

# --- (核心参数) “长光变换”的参数 ---
# 请将您软件截图中的参数填入此处
CHANGGUANG_PARAMS = {
    'alpha': 0.67,
    'beta': 0.75,
    'X': 0.025,
    'Y': 0.005
}

# --- “百分比截断线性拉伸”的参数 (当 STRETCH_METHOD 设置为 'percentile_stretch' 时生效) ---
PERCENTILE_STRETCH = (2.0, 98.0)


# =================================================================================


def changguang_transform(image_16bit, alpha, beta, x, y):
    """
    模拟并实现“长光变换”降位算法。
    这是一个基于改进的Gamma校正的非线性变换。
    """
    # 1. 归一化: 首先将16位数据归一化到 0.0 - 1.0 的浮点数范围
    # 为了避免极端值影响，我们先找到一个合理的拉伸范围 (例如1%到99%)
    min_val = np.percentile(image_16bit, 1.0)
    max_val = np.percentile(image_16bit, 99.0)
    if max_val <= min_val: max_val = min_val + 1 # 避免除以零

    # 进行截断和归一化
    normalized_image = (np.clip(image_16bit, min_val, max_val) - min_val) / (max_val - min_val)

    # 2. 应用核心变换公式
    # 这个公式是Gamma校正的一个变种，可以很好地模拟UI中的参数效果
    # output = alpha * (input + x)^beta + y
    transformed_image = alpha * np.power(normalized_image + x, beta) + y

    # 3. 转换回8位: 将结果拉伸到 0-255 并转换为8位整数
    # 再次进行截断以确保值在 0-1 范围内
    transformed_image = np.clip(transformed_image, 0.0, 1.0)
    image_8bit = (transformed_image * 255).astype(np.uint8)

    return image_8bit


def percentile_stretch_transform(image_16bit, lower_percent, upper_percent):
    """
    使用百分比截断线性拉伸算法将16位图像转换为8位图像。
    """
    image_8bit = np.zeros_like(image_16bit, dtype=np.uint8)
    for i in range(image_16bit.shape[2]):
        channel_16bit = image_16bit[:, :, i]
        min_val = np.percentile(channel_16bit, lower_percent)
        max_val = np.percentile(channel_16bit, upper_percent)
        if max_val == min_val:
            image_8bit[:, :, i] = 0
            continue
        clipped_channel = np.clip(channel_16bit, min_val, max_val)
        stretched_channel = 255 * (clipped_channel.astype(np.float32) - min_val) / (max_val - min_val)
        image_8bit[:, :, i] = stretched_channel.astype(np.uint8)
    return image_8bit


def process_directory(input_dir, output_dir):
    """批量处理指定目录下的所有高位深图像。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    supported_formats = ['.tif', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_formats]

    if not image_files:
        print(f"错误: 在输入文件夹 '{input_dir}' 中没有找到支持的图像文件 (.tif, .tiff)。")
        return

    print(f"找到 {len(image_files)} 张高位深图像，使用 '{STRETCH_METHOD}' 方法进行降位处理...")

    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_dir, filename)

        try:
            img_16bit = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img_16bit is None: raise IOError("无法解码图像数据")

            if img_16bit.ndim < 3 or img_16bit.shape[2] < 3:
                 print(f"\n警告: 图像 '{filename}' 不是多通道彩色图像，已跳过。")
                 continue
            if img_16bit.shape[2] > 3: img_16bit = img_16bit[:, :, :3]

            # 根据选择的方法执行降位
            if STRETCH_METHOD == 'changguang_transform':
                params = CHANGGUANG_PARAMS
                img_8bit = changguang_transform(img_16bit, params['alpha'], params['beta'], params['X'], params['Y'])
            elif STRETCH_METHOD == 'percentile_stretch':
                img_8bit = percentile_stretch_transform(img_16bit, PERCENTILE_STRETCH[0], PERCENTILE_STRETCH[1])
            else:
                print(f"\n错误: 未知的降位方法 '{STRETCH_METHOD}'。请检查 STRETCH_METHOD 的值。")
                continue

            base_filename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_filename}_8bit.png")
            cv2.imencode('.png', img_8bit)[1].tofile(output_path)

        except Exception as e:
            print(f"\n处理文件 '{filename}' 时发生错误: {e}")

    print("\n所有图像降位处理完毕！")
    print(f"结果已保存在 '{output_dir}' 文件夹中。")


if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER_16BIT):
        print(f"错误: 输入文件夹 '{INPUT_FOLDER_16BIT}' 不存在。")
    else:
        process_directory(INPUT_FOLDER_16BIT, OUTPUT_FOLDER_8BIT)

