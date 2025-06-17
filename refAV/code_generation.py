from pathlib import Path
import os
import ast
import sys
from pathlib import Path
from typing import List, Optional, TextIO
import json
import time
import refAV.paths as paths
import re
import json

# API specific imports located within LLM-specific scenario prediction functions


def extract_and_save_code_blocks(message, description=None, output_dir:Path=Path('.'))->list[Path]:
    """
    Extracts Python code blocks from a message and saves them to files based on their description variables.
    Handles both explicit Python code blocks (```python) and generic code blocks (```).
    """
    # try:
    # Split the message into lines and handle escaped characters
    lines = message.replace('\\n', '\n').replace("\\'", "'").split('\n')
    in_code_block = False
    current_block = []
    code_blocks = []
    
    for line in lines:
        # Check for code block markers

        if line.strip().startswith('```'):
            # If we're not in a code block, start one
            if not in_code_block:
                in_code_block = True
                current_block = []
            # If we're in a code block, end it
            else:
                in_code_block = False
                if current_block:  # Only add non-empty blocks
                    code_blocks.append('\n'.join(current_block))
                current_block = []
            continue
            
        # If we're in a code block, add the line
        if in_code_block:
            # Skip the "python" language identifier if it's there
            if line.strip().lower() == 'python':
                continue
            if 'description =' in line:
                continue

            current_block.append(line)
    # except:
        # import pdb;pdb.set_trace()
    # Process each code block
    filenames = []
    for i, code_block in enumerate(code_blocks):
        
        # Save the code block
        if description:
            filename = output_dir / f"{description}.txt"
        else:
            filename = output_dir / 'default.txt'
            
        try:
            with open(filename, 'w') as f:
                f.write(code_block)
            filenames.append(filename)
        except Exception as e:
            # import pdb;pdb.set_trace()
            print(f"Error saving file {filename}: {e}")

    return filenames


def predict_scenario_from_description(natural_language_description, output_dir:Path, 
        model_name:str='gemini-2.0-flash',
        local_model = None, local_tokenizer=None, destructive=False):
        
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    definition_filename = output_dir / (natural_language_description + '.txt')

    # if definition_filename.exists() and not destructive:
    #     print(f'Cached scenario for description {natural_language_description} already found.')
    #     return definition_filename
    
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()
    with open(paths.AV2_CATEGORIES, 'r') as f:
        av2_categories = f.read().format()
    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()

    prompt = f"""
    Generate a Python code block that uses the provided functions to find instances of a referred object in an autonomous driving dataset, based on the given description. The code must adhere strictly to the description to avoid false positives.

    You are given the following context:
    - Context: {refav_context}
    - Categories: {av2_categories}
    - Natural language description: {natural_language_description}
    - Example output format: {prediction_examples}

    **Instructions:**
    - Output **only** a single Python code block (```python\n...\n```).
    - Include only the code and comments within the block.
    - Do not include any explanations, reasoning steps, or text outside the code block.
    - Do not define any additional functions, filepaths, or global variables. You may only use the predefined variables `log_dir` and `output_dir`.
    - **Always prioritize computationally efficient operations.** You must first filter objects by relative direction and/or distance before applying more complex pairwise or temporal comparisons. This is required to reduce computational overhead.
    - **If the provided functions are not expressive enough to fully implement the scenario**, still output the closest possible approximation using the provided functions, even if incomplete or approximate.
    - **Match the style, level of detail, and structure of the provided examples** when generating your output.

    Example of efficient filtering and structured output:

    ```python
    description="Find scenarios where the ego vehicle is following a vehicle being overtaken on the right within 25 meters"

    # Get ego vehicle and all vehicles
    ego_vehicle = get_objects_of_category(log_dir, category='EGO_VEHICLE')
    vehicles = get_objects_of_category(log_dir, category='VEHICLE')

    # First, find ego vehicle instances following any vehicle within 25 meters
    # This reduces the set of vehicles to process, improving efficiency
    vehicles_in_front = get_objects_in_relative_direction(
        ego_vehicle, vehicles, log_dir, direction='forward', within_distance=25
    )

    # Then, find vehicles being overtaken on the right among the filtered set
    # This avoids computing pairwise comparisons for all vehicles
    vehicles_being_overtaken = being_crossed_by(
        vehicles_in_front, vehicles, log_dir, direction='right'
    )

    # Output the scenario
    output_scenario(vehicles_being_overtaken, description, log_dir, output_dir)
    ```
    """


    if 'gemini' in model_name.lower():
        response = predict_scenario_gemini(prompt, model_name)
    elif 'qwen' in model_name.lower():
        response = predict_scenario_qwen(prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        response = predict_scenario_claude(prompt, model_name)

    try:
        definition_filename = extract_and_save_code_blocks(response, output_dir=output_dir, description=natural_language_description)[-1]
        #output_path = output_dir / (natural_language_description + '.txt')
        #with open(output_path, 'w') as file:
        #    file.write(response)
        print(f'{natural_language_description} definition saved to {output_dir}')
        return definition_filename
    except Exception as e:
        
        print(e)
        print(response)
        # import pdb;pdb.set_trace()
        print(f"Error saving description {natural_language_description}")
        return


def predict_scenario_from_description_global_context( model_name:str='gemini-2.0-flash'):
        
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()
    with open(paths.AV2_CATEGORIES, 'r') as f:
        av2_categories = f.read().format()
    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()
    with open(paths.PROMPT_CODE, 'r', encoding='utf-8') as f:
        natural_language_description = json.load(f)

    prompt = f"""
    You are an expert autonomous driving dataset mining agent.
    Your task is to generate a Python code block that uses the provided API functions to find instances of a referred object in an autonomous driving dataset, based on the given description.
    The code must strictly adhere to the description to avoid false positives.
    Categories:
    {av2_categories}

    Here is the API functions:
    {refav_context}

    Example output format:
    {prediction_examples}

    The file given to you contains all the descriptions to be processed. Please give me all the results of processing into code and save them in the format of a json file. Key is description and value is code:
    {natural_language_description}
    """


    if 'gemini' in model_name.lower():
        response = predict_scenario_gemini(prompt, model_name)
    elif 'qwen' in model_name.lower():
        response = predict_scenario_qwen(prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        response = predict_scenario_claude(prompt, model_name)
    
    return response

def refine_scenario_from_description_global_context(model_name:str='gemini-2.0-flash'):
        
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()
    with open(paths.AV2_CATEGORIES, 'r') as f:
        av2_categories = f.read().format()
    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()
    with open(paths.UNIQUE_PROMPT, 'r', encoding='utf-8') as f:
        log_language_pairs = json.load(f)
    prompt = f"""
    You are an expert autonomous driving dataset mining agent.
    Your task is to generate a Python code block that uses the provided API functions to find instances of a referred object in an autonomous driving dataset, based on the given description.
    The code must strictly adhere to the description to avoid false positives.
    Categories:
    {av2_categories}

    Here is the API functions:
    {refav_context}

    Example output format:
    {prediction_examples}

    The file given to you contains all the descriptions to be processed, as well as the results generated by a model ("description": "code" format). Please compare the results of the model corresponding to each description according to your own reasoning (you first reason and then analyze the results), and correct what you think is wrong. Save it in the format of a json file, the key is your newly generated code:
    {log_language_pairs}
    """

    if 'gemini' in model_name.lower():
        response = predict_scenario_gemini(prompt, model_name)
    elif 'qwen' in model_name.lower():
        response = predict_scenario_qwen(prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        response = predict_scenario_claude(prompt, model_name)
   
    return response


def predict_scenario_gemini(prompt, model_name):
    from google import genai
    """
    Available models:
    gemini-2.5-flash-preview-04-17
    gemini-2.0-flash
    """

    time.sleep(6)  #Free API limited to 10 requests per minute
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    config = {
        "temperature":0.8,
        "max_output_tokens":16000,
    }

    response = client.models.generate_content(
        model=model_name, contents=prompt, config=config
    )

    return response.text

def predict_scenario_claude(prompt, model_name):
    import anthropic
    
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        #api_key="my_api_key",
    )

    message = client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=.5,
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                    }
                ]
            }
        ]
    )

    # Convert the message content to string
    if hasattr(message, 'content'):
        content = message.content
    else:
        raise ValueError("Message object doesn't have 'content' attribute")
    
    if hasattr(content[0], 'text'):
        text_response = content[0].text
    elif isinstance(content, list):
        text_response = '\n'.join(str(item) for item in content)
    else:
        text_response = str(content)
        
    return text_response


# def load_qwen(model_name='Qwen2.5-7B-Instruct'):

#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch

#     qwen_model_name = "Qwen/" + model_name
#     model = AutoModelForCausalLM.from_pretrained(
#         qwen_model_name,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

#     return model, tokenizer

def load_qwen(model_name='Qwen3-0.6B'):
    """
    从本地路径加载Qwen模型和tokenizer
    
    参数:
        model_name: 模型名称，默认为'Qwen3-0.6B'
    返回:
        model: 加载的模型
        tokenizer: 对应的tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from pathlib import Path

    # 设置本地模型路径
    model_path = Path('../Qwen/Qwen3-0.6B')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 设置为评估模式
    model.eval()
    
    return model, tokenizer


def predict_scenario_qwen(prompt, model=None, tokenizer=None):

    if model == None or tokenizer == None:
        model, tokenizer = load_qwen()

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


class FunctionInfo:
    """Holds extracted information for a single function."""
    def __init__(self, name: str, signature_lines: List[str], docstring: Optional[str], col_offset: int):
        self.name = name
        # Keep signature as lines to preserve original formatting/indentation
        self.signature_lines = signature_lines
        self.docstring = docstring
        self.col_offset = col_offset # Store the column offset of the 'def' keyword

    def format_for_output(self) -> str:
        """Formats the function signature and docstring for display, including triple quotes."""
        # Determine base indentation from the 'def' line's column offset
        base_indent = " " * self.col_offset
        # Assume standard 4-space indentation for the body/docstring relative to the 'def' line
        body_indent = base_indent + "    "

        # Start with the signature lines
        # Strip trailing whitespace but keep leading whitespace (which is the base_indent)
        output_lines = [line.rstrip() for line in self.signature_lines]

        if self.docstring is not None:
            # Split the raw docstring content by lines
            docstring_lines = self.docstring.splitlines()

            # Add opening quotes line indented by body_indent
            output_lines.append(f"{body_indent}\"\"\"")

            # Add the docstring content lines, each indented by body_indent
            # ast.get_docstring already removes the *minimal* indentation from the *content block*.
            # So we just need to add the *body indent* to each line of the processed content.
            for line in docstring_lines:
                 output_lines.append(f"{body_indent}{line}")

            # Add closing quotes line indented by body_indent
            output_lines.append(f"{body_indent}\"\"\"")

        # Join the lines
        return "\n".join(output_lines).strip()


# --- AST Visitor to extract Function Info ---

class FunctionDocstringExtractor(ast.NodeVisitor):
    """AST visitor to find function definitions and extract their info."""
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        # Update the type hint for extracted_info to reflect the modified FunctionInfo
        self.extracted_info: List[FunctionInfo] = []

    def visit_FunctionDef(self, node):
        """Visits function definitions (def)."""
        name = node.name

        # Get the docstring using the standard ast helper
        docstring_content = ast.get_docstring(node)

        # Get the column offset of the 'def' keyword
        col_offset = node.col_offset

        # Determine the line number where the function body actually starts.
        body_start_lineno = node.lineno + 1
        if node.body:
            first_body_node = node.body[0]
            body_start_lineno = first_body_node.lineno

        # Extract signature lines: from the line of 'def' up to the line before the body starts.
        signature_lines_raw = self.source_lines[node.lineno - 1 : body_start_lineno - 1]

        # Pass the col_offset when creating the FunctionInfo object
        self.extracted_info.append(FunctionInfo(name, signature_lines_raw, docstring_content, col_offset))

        # We still don't generically visit children unless you uncomment generic_visit
        # self.generic_visit(node) # Keep commented unless you need nested functions/classes

    def visit_AsyncFunctionDef(self, node):
        """Visits async function definitions (async def)."""
        # Call the same logic as visit_FunctionDef
        self.visit_FunctionDef(node)


# --- Main Parsing Function ---

def parse_python_functions_with_docstrings(file_path: Path, output_path:Path) -> List[FunctionInfo]:
    """
    Parses a Python file to extract function definitions (signature) and their docstrings,
    excluding decorators.

    Args:
        file_path: Path to the Python file.

    Returns:
        A list of FunctionInfo objects, each containing the function name,
        signature lines (without decorators), and docstring. Returns an empty
        list in case of errors.
    """
    try:
        # Read the file content, specifying encoding for robustness
        source_code = file_path.read_text(encoding='utf-8')
        # Keep original lines to reconstruct signatures
        lines = source_code.splitlines()

        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)

        # Use the visitor to walk the tree and extract info
        visitor = FunctionDocstringExtractor(lines)
        visitor.visit(tree) # Start the traversal

        with open(output_path, 'w') as file:
            display_function_info(visitor.extracted_info, file)

        return visitor.extracted_info

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}", file=sys.stderr)
        return []


def display_function_info(function_info_list: List[FunctionInfo], output_stream: TextIO = sys.stdout):
    """
    Displays the extracted function information (signature and docstring)
    to the specified output stream in the requested text format.

    Args:
        function_info_list: A list of FunctionInfo objects.
        output_stream: The stream to write the output to (e.g., sys.stdout, a file object).
    """
    for i, func_info in enumerate(function_info_list):
        if i > 0:
            # Add a separator between function outputs for clarity, matching the previous output
            output_stream.write("\n\n")

        # Use the format_for_output method to get the combined signature and docstring
        formatted_text = func_info.format_for_output()
        output_stream.write(formatted_text)
        output_stream.write("\n") # Ensure a newline after each function block

def build_plan_prompt(description, refav_context, av2_categories):
    prompt = f"""
    You are an expert planner for autonomous driving scene mining.

    Your task is to write a clear, step-by-step plan for how to implement the following scenario:

    "{description}"

    You must use the following API functions:

    {refav_context}

    Categories: 
    
    {av2_categories}

    Your plan should cover:

    0. What is the overall reasoning goal to identify the described scenario? (one concise sentence that summarizes the intent)

    1. What objects need to be queried? (list object categories to retrieve)

    2. What filtering should be applied to reduce candidate objects? (describe coarse filtering steps to apply)

    3a. What spatial relations need to be computed (including any use of relative direction)? (describe spatial relations and relevant API calls)

    3b. What temporal relations need to be computed (if applicable)? (describe temporal relations and relevant API calls)

    4. How should the results be combined? (describe the logical combination of results)

    5. How should the scenario be output? (describe the final output step)

    Output the plan as clearly numbered steps (Step 0, Step 1, ..., Step 5).  
    Do not write any Python code in this step.  
    Do not output anything other than the numbered plan.
    """
    return prompt


def build_code_from_plan_prompt(description, plan_text, prediction_examples, refav_context):
    prompt = f"""
    You are an expert autonomous driving dataset mining agent.

    You are given the following step-by-step plan to implement the scenario:

    {plan_text}

    Now, generate the Python code block to implement this plan, using the provided API functions:

    {refav_context}

    Your code must strictly follow the provided example style.

    In the code, your comments must reflect your reasoning steps clearly and concisely, aligned with the plan steps. This acts as a Chain-of-Thought reasoning trace.

    Do not output any text outside the Python code block.

    Example correct code style:

    {prediction_examples}

    Now, implement the following scenario:

    "{description}"
    """
    return prompt

def predict_scenario_from_description_plan(natural_language_description, output_dir:Path, 
        model_name:str='gemini-2.0-flash',
        local_model = None, local_tokenizer=None, destructive=False):
        
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    definition_filename = output_dir / (natural_language_description + '.txt')

    # if definition_filename.exists() and not destructive:
    #     print(f'Cached scenario for description {natural_language_description} already found.')
    #     return definition_filename
    
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()
    with open(paths.AV2_CATEGORIES, 'r') as f:
        av2_categories = f.read().format()
    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()

    plan_prompt = build_plan_prompt(natural_language_description, refav_context, av2_categories)

    if 'gemini' in model_name.lower():
        plan_text = predict_scenario_gemini(plan_prompt, model_name)
    elif 'qwen' in model_name.lower():
        plan_text = predict_scenario_qwen(plan_prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        plan_text = predict_scenario_claude(plan_prompt, model_name)

    code_prompt = build_code_from_plan_prompt(natural_language_description, plan_text, prediction_examples, refav_context)
    if 'gemini' in model_name.lower():
        response = predict_scenario_gemini(code_prompt, model_name)
    elif 'qwen' in model_name.lower():
        response = predict_scenario_qwen(code_prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        response = predict_scenario_claude(code_prompt, model_name)
    try:
        definition_filename = extract_and_save_code_blocks(response, output_dir=output_dir, description=natural_language_description)[-1]
        #output_path = output_dir / (natural_language_description + '.txt')
        #with open(output_path, 'w') as file:
        #    file.write(response)
        print(f'{natural_language_description} definition saved to {output_dir}')
        return definition_filename
    except Exception as e:
        
        print(e)
        print(response)
        # import pdb;pdb.set_trace()
        print(f"Error saving description {natural_language_description}")
        return

def build_reranker_prompt(description, example_code, candidates,api_func,av2_categories):
    prompt = f"""
    You are an expert code reviewer for autonomous driving dataset mining tasks.

    You are given multiple candidate Python code blocks generated by different models. Your job is to carefully select the single BEST code block, according to the following strict criteria:

    Evaluation Criteria:
    1. The code must match the example style, indentation, and comment level.
    2. The code must use only the provided API functions correctly:
    {api_func}
    
    3. The code must be logically correct and complete for the given task description.
    4. Refer to the object classes:
    {av2_categories}

    Example correct code:

    {example_code}

    ---

    Task Description:
    "{description}"

    ---

    Candidate Code Blocks:

    """

    for idx, cand in enumerate(candidates):
        prompt += f"\nCandidate #{idx+1} (from model {cand['model']}):\n```python\n{cand['code']}\n```"

    prompt += """

    ---

    Now, based on the criteria, select the BEST candidate.

    Respond in the following format:

    Best Candidate #: <number>

    Explanation: <explanation of why you selected this candidate and why the others were not chosen.>

    Do not output any other text.
    """

    return prompt

def build_reranker_prompt_with_metric(description, example_code, candidates_with_metrics,api_func,av2_categories):
    prompt = f"""
    You are an expert code reviewer for autonomous driving dataset mining tasks.

    You are given multiple candidate Python code blocks generated by different models, along with their execution metrics.

    Your job is to carefully select the single BEST code block, according to the following strict criteria:

    Primary Evaluation Criteria (most important):
    1. The code must match the example style, indentation.
    2. The code must use only the provided API functions correctly:
    {api_func}
    
    3. The code must be logically correct and complete for the given task description.
    4. Refer to the object classes:
    {av2_categories}

    Secondary Evaluation Criteria (additional signals to consider, but not primary factors):
    - Execution Metric Rank (lower rank is better).

    Example correct code:

    {example_code}

    ---

    Task Description:
    "{description}"

    ---

    Candidate Code Blocks:

    """

    for idx, cand in enumerate(candidates_with_metrics):
        prompt += f"Execution Metric Rank: {cand['metric_rank']} / {len(candidates_with_metrics)}\n"
        prompt += f"```python\n{cand['code']}\n```\n"

    prompt += """

    ---

    Now, based on the above criteria, select the BEST candidate.

    Respond in the following format:

    Best Candidate #: <number>

    Overall Ranking: X > Y > Z

    Explanation: <explanation of why you selected this candidate, how you considered both the primary and secondary criteria, and why the others were not chosen.>

    Do not output any other text.
    """

    return prompt


def predict_scenario_from_description_voting(natural_language_description, output_dir:Path, 
        model_name:str='gemini-2.0-flash',
        local_model = None, local_tokenizer=None, destructive=False):
        
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    definition_filename = output_dir / (natural_language_description + '.txt')

    # if definition_filename.exists() and not destructive:
    #     print(f'Cached scenario for description {natural_language_description} already found.')
    #     return definition_filename
    
    with open(paths.REFAV_CONTEXT, 'r') as f:
        refav_context = f.read().format()
    with open(paths.AV2_CATEGORIES, 'r') as f:
        av2_categories = f.read().format()
    with open(paths.PREDICTION_EXAMPLES, 'r') as f:
        prediction_examples = f.read().format()

    with open(os.path.join('/mnt/zhenghuan/RefAV/RefAV/output/llm_code_predictions/gemini-2.5-pro-plan',(natural_language_description + '.txt')), 'r') as f:
        candidate1_code = f.read().format()
    with open(os.path.join('/mnt/zhenghuan/RefAV/RefAV/output/llm_code_predictions/gemini-2.5-pro-preview-06-05', (natural_language_description + '.txt')), 'r') as f:
        candidate2_code = f.read().format()
    with open(os.path.join('/mnt/zhenghuan/RefAV/RefAV/output/llm_code_predictions/claude-opus-4-20250514',(natural_language_description + '.txt')), 'r') as f:
        candidate3_code = f.read().format()
    candidate1={'code':candidate1_code,'metric_rank':2}
    candidate2={'code':candidate2_code,'metric_rank':1}
    candidate3={'code':candidate3_code,'metric_rank':3}


    candidates=[candidate1,candidate2,candidate3]

    voting_prompt = build_reranker_prompt_with_metric(natural_language_description, prediction_examples, candidates,refav_context,av2_categories)
    
    

    if 'gemini' in model_name.lower():
        response = predict_scenario_gemini(voting_prompt, model_name)
    elif 'qwen' in model_name.lower():
        response = predict_scenario_qwen(voting_prompt, local_model, local_tokenizer)
    elif 'claude' in model_name.lower():
        response = predict_scenario_claude(voting_prompt, model_name)


    try:
        # text = response["choices"][0]["message"]["content"].strip()
        text=response
        match = re.search(r"Best Candidate #:\s*(\d+)", text)
        ranking_match = re.search(r"Overall Ranking:\s*(.*)", text)
        explanation_match = re.search(r"Explanation:\s*(.*)", text, re.DOTALL)

        explanation = explanation_match.group(1).strip() if explanation_match else ""
        # import pdb;pdb.set_trace()
        if match:
            best_idx = int(match.group(1)) - 1
            if 0 <= best_idx < len(candidates):
                print(f"✅ Reranker selected Candidate #{best_idx+1} ")
                # print(f"Explanation: {explanation}")
                response= candidates[best_idx]['code']
                
            else:
                print("⚠️ Reranker output index out of range.")
                response= None
        else:
            print("⚠️ Reranker output format invalid.")
            response= None

        # definition_filename = extract_and_save_code_blocks(response, output_dir=output_dir, description=natural_language_description)[-1]
        output_path = output_dir / (natural_language_description + '.txt')
        with open(output_path, 'w') as file:
           file.write(response)
        print(f'{natural_language_description} definition saved to {output_dir}')
        return definition_filename
    except Exception as e:
        
        print(e)
        print(response)
        print(f"Error saving description {natural_language_description}")
        return
if __name__ == '__main__':

    atomic_functions_path = Path('/home/crdavids/Trinity-Sync/refbot/refAV/atomic_functions.py')
    parse_python_functions_with_docstrings(atomic_functions_path, paths.REFAV_CONTEXT)

    all_descriptions = set()
    with open('av2_sm_downloads/log_prompt_pairs_val.json', 'rb') as file:
        lpp_val = json.load(file)

    with open('av2_sm_downloads/log_prompt_pairs_test.json', 'rb') as file:
        lpp_test = json.load(file)

    for log_id, prompts in lpp_val.items():
        all_descriptions.update(prompts)
    for log_id, prompts in lpp_test.items():
        all_descriptions.update(prompts)

    print(len(all_descriptions))

    output_dir = paths.LLM_PRED_DIR / 'claude-3-7-sonnet-20250219'
    for description in all_descriptions:
        #break
        #predict_scenario_from_description(description, output_dir, model_name='claude-3-5-sonnet-20241022')
        predict_scenario_from_description(description, output_dir, model_name='claude-3-7-sonnet-20250219')
        #predict_scenario_from_description(description, output_dir, model_name='gemini-2.5-flash-preview-04-17')

        #predict_scenario_from_description(description, output_dir, model_name='gemini-2.0-flash')


        
    
    #model_name = 'Qwen2.5-7B-Instruct'
    #model_name = 'Qwen3-32B'
    #local_model, local_tokenizer = load_qwen(model_name)
    #for description in all_descriptions:
        #predict_scenario_from_description(description, output_dir, model_name=model_name, local_model=local_model, local_tokenizer=local_tokenizer)



    