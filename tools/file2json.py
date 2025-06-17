import os
import json

def aggregate_to_json(input_prefix, output_json_path):
    # 初始化结果字典
    data = {}
    
    # 遍历输入前缀下的所有文件夹
    for folder_name in os.listdir(input_prefix):
        folder_path = os.path.join(input_prefix, folder_name)
        # import pdb;pdb.set_trace()
        # 确保是文件夹
        # if os.path.isdir(folder_path):
        # 构造预期的 .txt 文件路径（文件名与文件夹名相同）
        txt_file_path = folder_path
        
        # 检查文件是否存在
        if os.path.isfile(txt_file_path):
            # 读取文件内容
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用文件夹名作为 key，文件内容作为 value
            # 将文件夹名中的下划线转换回空格（假设原始 key 使用空格）
            # key = folder_name.replace('_', ' ')
            key=folder_name.replace('.txt', '')
            data[key] = content
    
    # 将结果写入 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    input_prefix = "output/llm_code_predictions/gemini-2.5-pro-voting"  # 替换为你的输入路径前缀
    output_json_path = "results/gemini-2.5-pro-voting.json"  # 替换为你想要的输出 JSON 文件路径
    aggregate_to_json(input_prefix, output_json_path)
    print("JSON 文件已成功创建！")