import pickle

def read_pkl_file():
    try:
        with open('combined_predictions_val.pkl', 'rb') as f:
            data = pickle.load(f)
            
        target_log_id = '96dd6923-994c-3afe-9830-b15bdfd60f64'
        found = False
        
        print(f"查找log_id为 {target_log_id} 的所有prompt：")
        print("-" * 50)
        
        # 遍历数据查找匹配的log_id
        for (log_id, prompt), _ in data.items():
            if log_id == target_log_id:
                found = True
                print(f"prompt: {prompt}")
                
        if not found:
            print(f"未找到log_id为 {target_log_id} 的数据")
        
    except FileNotFoundError:
        print("找不到文件 combined_predictions_val.pkl")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")

def analyze_prompts():
    try:
        with open('/output/sm_predictions/exp_claude4.0/results/combined_predictions_test.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # 统计每个log_id的prompt数量
        log_id_prompts = {}
        for (log_id, prompt), _ in data.items():
            if log_id not in log_id_prompts:
                log_id_prompts[log_id] = set()
            log_id_prompts[log_id].add(prompt)
        
        # 计算统计信息
        prompt_counts = [len(prompts) for prompts in log_id_prompts.values()]
        total_log_ids = len(log_id_prompts)
        total_combinations = sum(prompt_counts)
        max_prompts = max(prompt_counts)
        min_prompts = min(prompt_counts)
        
        print("\n统计信息：")
        print("-" * 50)
        print(f"总共有 {total_log_ids} 个不同的log_id")
        print(f"总共有 {total_combinations} 个log_id+prompt组合")
        print(f"每个log_id最多有 {max_prompts} 个不同的prompt")
        print(f"每个log_id最少有 {min_prompts} 个不同的prompt")
        
        # 打印一些示例
        print("\n示例：")
        print("-" * 50)
        print("拥有最多prompt的log_id示例：")
        for log_id, prompts in log_id_prompts.items():
            if len(prompts) == max_prompts:
                print(f"log_id: {log_id}")
                print(f"prompt数量: {len(prompts)}")
                print("prompts:")
                for prompt in prompts:
                    print(f"  - {prompt}")
                break
        
    except FileNotFoundError:
        print("找不到文件 combined_predictions_val.pkl")
    except Exception as e:
        print(f"分析数据时发生错误: {str(e)}")

if __name__ == "__main__":
    read_pkl_file()
    print("\n" + "="*50 + "\n")
    analyze_prompts()
