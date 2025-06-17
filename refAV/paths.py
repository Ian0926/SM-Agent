from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('/mnt/dataset/argoverse2')
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('av2_sm_downloads')

# path to cached atomic function outputs, likely does not exist for you
CACHE_PATH = Path('/home/crdavids/Trinity-Sync/av2-api/output/misc')

#input directories, do not change
EXPERIMENTS = Path('run/experiments.yml')
REFAV_CONTEXT = Path('refAV/llm_prompting/refAV_context.txt')
AV2_CATEGORIES = Path('refAV/llm_prompting/av2_categories.txt')
PREDICTION_EXAMPLES = Path('refAV/llm_prompting/prediction_examples.txt')
PREDICTION_EXAMPLES_MINI = Path('refAV/llm_prompting/prediction_examples_mini.txt')
PREDICTION_EXAMPLES_MINI2 = Path('refAV/llm_prompting/prediction_examples_mini2.txt')
UNIQUE_PROMPT = Path('results_tmp/log_prompt_pairs_test_unique.json')
PROMPT_CODE = Path('results_tmp/prompt_code.json')
UNCOVERED_KEYS_REPORT = Path('results_tmp/uncovered_keys.json')

#output directories, do not change
SM_DATA_DIR = Path('output/sm_dataset')
SM_PRED_DIR = Path('output/sm_predictions')
LLM_PRED_DIR = Path('output/llm_code_predictions')
TRACKER_PRED_DIR = Path('output/tracker_predictions')

