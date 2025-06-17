import os
import json
import re
import argparse
import refAV.paths as paths

def create_folders_from_json(json_file_path, output_prefix):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建基础输出目录（如果不存在）
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    
    # 为每个条目创建文件夹和文件
    for key, content in data.items():
        # 将 key 转换为有效的文件名（替换非法字符）
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', key)
        # folder_path = os.path.join(output_prefix, safe_filename)
        folder_path=output_prefix
        
        # 创建文件夹（如果不存在）
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 创建并写入以 key 命名的 .txt 文件
        file_path = os.path.join(folder_path, f"{safe_filename}.txt")
        # 将内容按行分割并逐行写入
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in content.splitlines():
                f.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--exp_name", type=str, default='exp53', help="Enter the name of the experiment from experiments.yml you would like to run.")
    parser.add_argument("--MAX_RETRIES", type=int, default=10)
    args = parser.parse_args()

    with open(paths.EXPERIMENTS, 'rb') as file:
        config = yaml.safe_load(file)

    json_file_path=paths.PROMPT_CODE
    output_prefix=Path('output/llm_code_predictions')/config[args.exp_name]['LLM']

    create_folders_from_json(json_file_path, output_prefix)
    print("文件夹和文件已成功创建！")