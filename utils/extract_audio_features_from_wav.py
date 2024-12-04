import os
import subprocess

from tqdm import tqdm

# OpenSMILE 二进制文件的路径
opensmile_path = "utils/opensmile-3.0-linux-x64/bin/SMILExtract"
# 配置文件路径
config_path = "utils/opensmile-3.0-linux-x64/config/is09-13/IS10_paraling.conf"
# 输入文件夹路径（包含所有的wav文件）
input_folder = "data/ECF/videos/valid"
# 输出文件夹路径（保存提取的特征）
output_folder = "data/ECF/videos/valid/opensmile-features"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

def extract_opensmile_audio_features_from_wav(input_path, output_path):
     # 调用 OpenSMILE 提取特征
    command = [
        opensmile_path, 
        "-C", config_path, 
        "-I", input_path, 
        "-O", output_path
    ]
    # 忽略输出和错误消息
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, stdout=devnull, stderr=devnull)

if __name__ == "__main__":
    # 遍历文件夹中所有的 .wav 文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".wav"):
            # 完整的输入文件路径
            input_file = os.path.join(input_folder, filename)
            # 完整的输出文件路径（CSV 格式）
            output_file = os.path.join(output_folder, filename.replace(".wav", ".arff"))
            if os.path.exists(output_file):
                print(f"Skipping {filename}, output already exists at {output_file}")
                continue
            # 调用 OpenSMILE 提取特征
            command = [
                opensmile_path, 
                "-C", config_path, 
                "-I", input_file, 
                "-O", output_file
            ]
            
            # 忽略输出和错误消息
            with open(os.devnull, 'w') as devnull:
                subprocess.run(command, stdout=devnull, stderr=devnull)

            print(f"Processed {filename}, output saved to {output_file}")