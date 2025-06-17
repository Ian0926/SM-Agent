import subprocess
import multiprocessing
import math
import argparse
import os
import sys
import json
from pathlib import Path
import tempfile
import refAV.paths as paths

def run_parallel_eval(exp_name: str, log_prompts_path: Path, procs_per_task: int = 2):
    """
    Launches multiple eval.py processes in parallel.
    It determines which log_id/prompt pairs still need processing,
    divides them among tasks, and allocates available CPUs dynamically.
    
    # 中文说明：
    # 该函数用于并行启动多个eval.py进程
    # 它会确定哪些log_id/prompt对还需要处理
    # 将这些任务分配给不同的进程
    # 并动态分配可用的CPU资源
    """
    log_prompts_path = Path(log_prompts_path)

    print(f"Starting parallel evaluation for experiment: {exp_name}")
    print(f"Reading log prompts from: {log_prompts_path}")

    # Read the full log_prompts mapping
    # 读取完整的log_prompts映射文件
    with open(log_prompts_path, 'r') as file:
        lpp = json.load(file)

    # Determine the split name from the file name (assuming format like 'some_name_splitname.json')
    # 从文件名中获取数据集划分名称（假设格式为'some_name_splitname.json'）
    split_name = log_prompts_path.stem.split('_')[-1]

    # Build the list of (log_id, prompt) pairs that need to be completed
    # 构建需要完成的(log_id, prompt)对列表
    lpp_to_complete = []
    print("Checking which log_id/prompt pairs need evaluation...")
    for log_id, prompts in lpp.items():
        for prompt in prompts:
            # Construct the expected prediction file path
            # Assuming SM_PRED_DIR / exp_name / split_name / log_id / prompt_predictions.pkl
            # 构建预期的预测文件路径
            # 假设路径格式为 SM_PRED_DIR / exp_name / split_name / log_id / prompt_predictions.pkl
            pred_path = paths.SM_PRED_DIR / exp_name / split_name / log_id / f'{prompt}_predictions.pkl'
            if not pred_path.exists():
                lpp_to_complete.append((log_id, prompt))
    # lpp_to_complete=lpp_to_complete[::-1]
    print(lpp_to_complete)
    total_work_items = len(lpp_to_complete)
    print(f"Total log_id/prompt pairs requiring evaluation: {total_work_items}")
    # import pdb;pdb.set_trace()

    if total_work_items == 0:
        print("No evaluation needed. All predictions found.")
        return

    # Get available CPU count
    # 获取可用CPU数量
    cpu_count = multiprocessing.cpu_count()
    # Leave one CPU free for the parent script and other system processes
    # 保留一个CPU给父脚本和其他系统进程
    cpus_available_for_tasks = max(1, int(.95*(cpu_count)))

    print(f"System has {cpu_count} CPUs. {cpus_available_for_tasks} available for tasks.")
    print(f"Each task requests a base of {procs_per_task} processes.")

    # Calculate the maximum number of tasks we can run based on available CPUs and base procs per task
    # 根据可用CPU和每个任务的基础进程数计算最大并行任务数
    max_parallel_tasks_cpu_constrained = cpus_available_for_tasks // procs_per_task

    # The actual number of tasks is limited by the work items available and CPU capacity
    # 实际任务数受工作项数量和CPU容量的限制
    actual_num_tasks = min(total_work_items, max_parallel_tasks_cpu_constrained if max_parallel_tasks_cpu_constrained > 0 else 1)

    print(f"Calculated max parallel tasks based on CPUs ({cpus_available_for_tasks} / {procs_per_task}): {max_parallel_tasks_cpu_constrained}")
    print(f"Actual number of parallel tasks to launch (min of work and max_cpu): {actual_num_tasks}")

    if actual_num_tasks == 0: # Should not happen if total_work_items > 0, but for safety
    # 如果total_work_items > 0，这种情况不应该发生，但为了安全起见
         print("No tasks to launch.")
         return

    # --- CPU Allocation ---
    # Base processes per task
    # 每个任务的基础进程数
    base_procs_per_task = procs_per_task
    # Total CPUs needed if each task got the base number
    # 如果每个任务都获得基础进程数，所需的总CPU数
    total_base_procs_needed = actual_num_tasks * base_procs_per_task
    # Remaining CPUs to distribute
    # 剩余可分配的CPU数
    extra_cpus = cpus_available_for_tasks - total_base_procs_needed

    # Distribute extra CPUs one by one to the first tasks
    # 将额外CPU逐个分配给前面的任务
    task_procs_allocation = [base_procs_per_task] * actual_num_tasks
    for i in range(extra_cpus):
        task_procs_allocation[i % actual_num_tasks] += 1 # Cycle through tasks to add extra CPU
        # 循环分配额外CPU

    print(f"\nCPU allocation per task: {task_procs_allocation} (Total allocated: {sum(task_procs_allocation)} out of {cpus_available_for_tasks})")

    # --- Work Distribution (lpp_to_complete) ---
    # 工作分配
    chunk_size_work = math.ceil(total_work_items / actual_num_tasks)
    print(f"Total work items: {total_work_items}. Chunk size per task: {chunk_size_work}")

    processes = []
    temp_files = [] # List to store temporary file paths for cleanup
    # 存储临时文件路径的列表，用于清理
    print("\nStarting parallel tasks...")

    try:
        # Loop and run tasks in parallel
        # 循环并行运行任务
        for i in range(actual_num_tasks):
            # Get the work chunk for this task
            # 获取此任务的工作块
            start_index = i * chunk_size_work
            end_index = min(start_index + chunk_size_work, total_work_items)

            # Should not be empty based on actual_num_tasks calculation, but safe check
            # 基于actual_num_tasks的计算，不应该为空，但为了安全起见进行检查
            if start_index >= end_index:
                continue

            task_work_tuples = lpp_to_complete[start_index:end_index]

            # Convert list of tuples to the required dictionary format for the temporary file
            # 将元组列表转换为临时文件所需的字典格式
            task_lpp_dict = {}
            for log_id, prompt in task_work_tuples:
                if log_id not in task_lpp_dict:
                    task_lpp_dict[log_id] = []
                task_lpp_dict[log_id].append(prompt)

            # Create a temporary JSON file for this task's work
            # use delete=False so the file persists after closing, allowing subprocess access
            # 为此任务的工作创建临时JSON文件
            # 使用delete=False使文件在关闭后仍然存在，允许子进程访问
            try:
                temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', prefix=f'task_{i}_')
                temp_file_path = Path(temp_file.name)
                temp_files.append(temp_file_path)

                # Write the task's work dictionary to the temporary file
                # 将任务的工作字典写入临时文件
                json.dump(task_lpp_dict, temp_file, indent=4)
                temp_file.close() # Close the file so the subprocess can open it
                # 关闭文件以便子进程可以打开它

            except Exception as e:
                 print(f"Error creating or writing to temporary file for task {i+1}: {e}", file=sys.stderr)
                 # Clean up files created so far before exiting
                 # 在退出前清理已创建的文件
                 raise # Re-raise the exception to jump to the finally block
                 # 重新抛出异常以跳转到finally块

            current_procs = task_procs_allocation[i]
            print(f"  Launching task {i+1}/{actual_num_tasks}: processing {len(task_work_tuples)} pairs, using {current_procs} processes. Temp file: {temp_file_path}")

            # Construct the command and arguments for the subprocess
            # 构建子进程的命令和参数
            command = [
                sys.executable, # Use the same python interpreter that is running this script
                # 使用运行此脚本的相同python解释器
                str(Path("refAV/eval.py")), # Ensure path is correct
                # 确保路径正确
                "--exp_name", exp_name,
                "--log_prompt_pairs", str(temp_file_path), # Pass the path to the temporary file
                # 传递临时文件的路径
                "--num_processes", str(current_procs) # Pass the allocated number of processes
                # 传递分配的进程数
            ]

            try:
                # Use subprocess.Popen to run the command in the background
                # Leaving stdout=None and stderr=None (the default) means
                # the subprocess's output will go to the parent process's stdout/stderr,
                # which is typically your terminal.
                # 使用subprocess.Popen在后台运行命令
                # 将stdout和stderr设为None（默认值）意味着
                # 子进程的输出将发送到父进程的stdout/stderr，
                # 通常是您的终端
                process = subprocess.Popen(command)
                processes.append((process, i, temp_file_path)) # Store process object and info
                # 存储进程对象和信息

            except FileNotFoundError:
                print(f"Error: Python interpreter '{sys.executable}' or script 'refAV/eval.py' not found. Make sure you are running from the project root.", file=sys.stderr)
                # Terminate any processes already started if a critical error occurs
                # 如果发生严重错误，终止已启动的进程
                for p, _, _ in processes:
                    try:
                        p.terminate()
                    except ProcessLookupError:
                        pass # Process already finished
                        # 进程已经结束
                raise # Jump to finally block for cleanup
                # 跳转到finally块进行清理

            except Exception as e:
                print(f"An error occurred launching task {i+1}: {e}", file=sys.stderr)
                 # Terminate any processes already started
                 # 终止已启动的进程
                for p, _, _ in processes:
                     try:
                        p.terminate()
                     except ProcessLookupError:
                         pass # Process already finished
                         # 进程已经结束
                raise # Jump to finally block for cleanup
                # 跳转到finally块进行清理


        print(f"\nWaiting for all {len(processes)} tasks to complete...")

        # Wait for all processes to finish and check their return codes
        # 等待所有进程完成并检查它们的返回码
        for process, task_index, temp_file_path in processes:
            try:
                return_code = process.wait() # Wait for this specific process to finish
                # 等待特定进程完成
                if return_code != 0:
                    print(f"\nWARNING: Task {task_index+1} (temp file: {temp_file_path}) failed with return code {return_code}", file=sys.stderr)
                else:
                    print(f"\nTask {task_index+1} (temp file: {temp_file_path}) completed successfully.")
            except Exception as e:
                 print(f"\nError waiting for task {task_index+1}: {e}", file=sys.stderr)

        print("\nAll parallel tasks finished.")

    finally:
        # Clean up temporary files
        # 清理临时文件
        print("\nCleaning up temporary files...")
        for temp_file_path in temp_files:
            try:
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                    print(f"  Removed {temp_file_path}")
                else:
                     print(f"  Temporary file not found (already removed?): {temp_file_path}")
            except Exception as e:
                print(f"  Error removing temporary file {temp_file_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Run parallel eval.py tasks.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Name of the experiment from the exp.yml file")
    parser.add_argument("--log_prompts_path", type=str, required=True, help="Path to the JSON file containing log_id to prompt list mapping.")
    parser.add_argument("--procs_per_task", type=int, default=2, help="Base number of processes to request for each eval.py task. Extra available CPUs will be distributed.")
    args = parser.parse_args()

    # 运行并行评估
    run_parallel_eval(
        exp_name=args.exp_name,
        log_prompts_path=args.log_prompts_path,
        procs_per_task=args.procs_per_task
    )