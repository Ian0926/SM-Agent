import time
import refAV.paths as paths
from refAV.code_generation import predict_scenario_from_description_global_context as predict_scenario_from_description
from refAV.code_generation import refine_scenario_from_description_global_context as refine_generated_code
import argparse


def check_keys_and_get_uncovered(list_of_strings: list, dictionary_data: dict):
    """
    Checks if all strings in a list are present as keys in a dictionary.
    Returns a list of the keys that were not found (uncovered).
    """
    required_keys = set(list_of_strings)
    available_keys = set(dictionary_data.keys())
    uncovered_items = sorted(list(required_keys - available_keys))
    return uncovered_items


def main(MODEL_NAME,MAX_RETRIES):
    """Main workflow function."""
    print("--- Starting SM-Agent Workflow ---")

    # 1. Load all required descriptions from the file
    try:
        with open(paths.UNIQUE_PROMPT, 'r', encoding='utf-8') as f:
            all_required_descriptions = json.load(f)
        print(f"Successfully loaded {len(all_required_descriptions)} descriptions.")
    except FileNotFoundError:
        print(f"Error: Input description file not found at: {paths.UNIQUE_PROMPT}")
        return
    
    # --- Generation and Iteration Phase ---
    generated_code = {}
    descriptions_to_process = all_required_descriptions
    
    for i in range(MAX_RETRIES):
        print(f"\n--- Code Generation Round {i + 1}/{MAX_RETRIES} ---")
        print(f"Processing {len(descriptions_to_process)} descriptions in this round.")
        
        if not descriptions_to_process:
            print("No more descriptions to process.")
            break

        # Call the generation function
        newly_generated_code = predict_scenario_from_description(descriptions_to_process, MODEL_NAME)
        
        
        print(f"Model successfully generated {len(newly_generated_code)} code snippets.")
        # Merge the newly generated code into the main dictionary
        generated_code.update(newly_generated_code)

        # Check for any descriptions that were not covered
        uncovered = check_keys_and_get_uncovered(all_required_descriptions, generated_code)
        
        if not uncovered:
            print("\nSuccess! Code has been generated for all descriptions.")
            break
        else:
            print(f"{len(uncovered)} descriptions remain uncovered. Preparing for the next round...")
            print("Uncovered descriptions:", uncovered)
            # Update the list for the next iteration to only include the missing items
            descriptions_to_process = uncovered
            time.sleep(1)
    
    # After the loop, do a final check
    final_uncovered = check_keys_and_get_uncovered(all_required_descriptions, generated_code)
    if final_uncovered:
        print("\n--- Workflow Warning ---")
        print(f"After reaching max retries, {len(final_uncovered)} descriptions still have no code.")
        print("Unfinished descriptions:", final_uncovered)
        with open(paths.UNCOVERED_KEYS_REPORT, 'w', encoding='utf-8') as f:
            json.dump(final_uncovered, f, indent=4)
        print(f"List of unfinished items saved to: {paths.UNCOVERED_KEYS_REPORT}")

    # --- Refinement Phase ---
    if not generated_code:
        print("No code was generated, skipping refinement. Workflow ends.")
        return
        
    print("\n--- Starting Code Refinement Phase ---")
    refined_code = refine_generated_code(generated_code, MODEL_NAME)

    
    with open(paths.PROMPT_CODE, 'w', encoding='utf-8') as f:
        json.dump(refined_code, f, indent=4, ensure_ascii=False)
    print(f"Final refined code saved to: {paths.PROMPT_CODE}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--exp_name", type=str, default='exp53', help="Enter the name of the experiment from experiments.yml you would like to run.")
    parser.add_argument("--MAX_RETRIES", type=int, default=10)
    args = parser.parse_args()

    with open(paths.EXPERIMENTS, 'rb') as file:
        config = yaml.safe_load(file)

    main(config[args.exp_name]['LLM'],args.MAX_RETRIES)