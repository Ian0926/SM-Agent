import json
import pickle
from itertools import product

# 读取 JSON 文件
def load_json_file(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

# 读取 PKL 文件
def load_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

# 生成所有可能的 (UUID, 描述) 组合
def generate_uuid_desc_pairs(json_data):
    pairs = []
    for uuid, desc_list in json_data.items():
        for desc in desc_list:
            pairs.append((uuid, desc))
    return set(pairs)  # 使用集合以便比较

# 找出 PKL 文件中缺失的 (UUID, 描述) 对
def find_missing_pairs(json_path, pkl_path):
    # 加载文件
    json_data = load_json_file(json_path)
    pkl_data = load_pkl_file(pkl_path)
    
    # 生成 JSON 文件中的所有 (UUID, 描述) 组合
    json_pairs = generate_uuid_desc_pairs(json_data)
    
    # 获取 PKL 文件中的键（假设为 (UUID, 描述) 元组）
    pkl_keys = set(pkl_data.keys())
    
    # 找出缺失的组合
    missing_pairs = json_pairs - pkl_keys
    
    return missing_pairs

# 主程序
def main():
    json_path = 'av2_sm_downloads/log_prompt_pairs_test.json'  # 替换为你的 JSON 文件路径
    pkl_path = 'sm_predictions/exp998/results/combined_predictions_test.pkl'    # 替换为你的 PKL 文件路径
    
    missing = find_missing_pairs(json_path, pkl_path)
    
    if missing:
        print("PKL 文件中缺失的 (UUID, 描述) 对：")
        for uuid, desc in missing:
            print(f"UUID: {uuid}, 描述: {desc}")
    else:
        print("PKL 文件没有缺失任何 (UUID, 描述) 对。")

if __name__ == "__main__":
    main()