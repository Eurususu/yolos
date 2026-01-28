import os
import random
import shutil
from tqdm import tqdm  # 这是一个进度条库，如果没有安装，可以去掉相关代码或运行 pip install tqdm

def extract_calibration_images(source_dir, target_dir, num_images, extensions=None):
    """
    从源目录随机抽取指定数量的图片复制到目标目录。
    """
    # 1. 设置默认支持的图片格式
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 确保扩展名都是小写，方便匹配
    extensions = [ext.lower() for ext in extensions]

    # 2. 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在！")
        return

    # 3. 获取源目录下所有符合格式的图片文件
    all_files = os.listdir(source_dir)
    image_files = [
        f for f in all_files 
        if os.path.splitext(f)[1].lower() in extensions
    ]

    total_images = len(image_files)
    print(f"源目录中共有 {total_images} 张图片。")

    # 4. 检查是否有足够的图片
    if total_images == 0:
        print("未找到任何图片文件。")
        return
    
    if num_images > total_images:
        print(f"警告: 请求抽取的数量 ({num_images}) 大于图片总数 ({total_images})。")
        print("将复制所有图片。")
        num_images = total_images

    # 5. 随机采样
    # 使用 random.sample 进行无放回抽样（不会抽到重复的）
    selected_images = random.sample(image_files, num_images)

    # 6. 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标目录已准备: {target_dir}")

    # 7. 开始复制
    print(f"开始复制 {num_images} 张图片...")
    
    # 如果安装了 tqdm，这里会显示进度条；如果没有，请改用普通 for 循环
    try:
        iterator = tqdm(selected_images)
    except NameError:
        iterator = selected_images

    for file_name in iterator:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        
        # copy2 会保留文件的元数据（如创建时间），适合数据集操作
        shutil.copy2(src_path, dst_path)

    print(f"\n成功！已将 {num_images} 张图片复制到 '{target_dir}'。")

# ==========================================
# 在这里修改你的路径配置
# ==========================================
if __name__ == "__main__":
    # 源图片目录（例如你的训练集图片目录）
    SOURCE_PATH = r"/home/jia/models/train" 
    
    # 新的校准集保存目录（脚本会自动创建）
    TARGET_PATH = r"/home/jia/models/calib_images"
    
    # 需要抽取的数量（通常 100 到 1000 之间足够）
    SAMPLE_NUM = 800 

    extract_calibration_images(SOURCE_PATH, TARGET_PATH, SAMPLE_NUM)