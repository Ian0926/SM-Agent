import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg') # Use a non-interactive backend for saving plots to files
import matplotlib.pyplot as plt
from pathlib import Path
from refAV.utils import get_log_split, swap_keys_and_listed_values
from tqdm import tqdm
import refAV.paths as paths
from refAV.atomic_functions import output_scenario, get_objects_of_category
import warnings
warnings.filterwarnings("ignore")


def eval_similarity_scores(
    input_json_path: str, 
    output_json_path: str, 
    majority_threshold: float = 0.75,
):
    """
    Processes a JSON file of tracker data, clusters confidence scores, selects tracks,
    and optionally visualizes the clusters.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file.
        k (int): The number of top clusters to select. If k <= 0, it defaults to 1.
        majority_threshold (float, optional): Min proportion of a track's scores in top k clusters. Defaults to 0.5.
        plot_output_dir (str, optional): Directory to save cluster visualizations. If None, no plots are saved.
    """
    output_json_path = Path(output_json_path)
    input_json_path = Path(input_json_path)
    #log_id_feather = {}

    # Load input JSON
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_json_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'.")
        return

    if not isinstance(data, dict):
        print(f"Error: Input JSON root must be a dictionary (object). Found type: {type(data)}.")
        return

    output_data = defaultdict(lambda: defaultdict(list))

    for prompt_id, logs in data.items():
        if not isinstance(logs, dict):
            warnings.warn(
                f"Warning: Content for prompt '{prompt_id}' is not a dictionary of logs. Skipping.",
                UserWarning
            )
            output_data[prompt_id] = {}
            continue

        all_confidence_scores_for_prompt = []
        track_confidences_map = {}

        if not logs:
            output_data[prompt_id] = {}
            continue

        for log_id, tracks_in_log in logs.items():
            #split = get_log_split(log_id)
            #if log_id not in log_id_feather:
            #    log_id_feather[log_id] = pd.read_feather(
            #        paths.TRACKER_PRED_DIR / (input_json_path.stem.split('_')[0] + '_' + input_json_path.stem.split('_')[1]) / split / log_id / 'sm_annotations.feather')

            output_data[prompt_id][log_id] = []
            if not isinstance(tracks_in_log, dict):
                warnings.warn(
                    f"Warning: Content for log '{log_id}' (prompt '{prompt_id}') is not a dict of tracks. Skipping.",
                    UserWarning
                )
                continue
            
            for track_uuid, timestamps in tracks_in_log.items():
                current_track_scores = []
                track_confidences_map[(log_id, track_uuid)] = []
                if not isinstance(timestamps, dict):
                    warnings.warn(
                        f"Warning: Timestamps for track '{track_uuid}' (log '{log_id}', prompt '{prompt_id}') "
                        f"is not a dictionary. Skipping scores for this track.",
                        UserWarning
                    )
                    continue
                
                #Reduces the amount of length 1 tracks that make it through to the final set
                modifier = -0.05/len(timestamps)
                modifier = 0
                for _, confidence in timestamps.items():
                    if isinstance(confidence, (int, float)):
                        all_confidence_scores_for_prompt.append(confidence + modifier)
                        current_track_scores.append(confidence + modifier)
                    else:
                        warnings.warn(
                            f"Warning: Invalid confidence value '{confidence}' (type: {type(confidence)}) "
                            f"for prompt '{prompt_id}', log '{log_id}', track '{track_uuid}'. Skipping.",
                            UserWarning
                        )
                track_confidences_map[(log_id, track_uuid)] = current_track_scores
        
        scores_np = np.array(all_confidence_scores_for_prompt)
        #Set the threshold at the 75%-ile 
        scores_np = np.sort(scores_np)
        threshold = scores_np[int(majority_threshold*(len(scores_np)))]
        for (log_id, track_uuid), track_scores in track_confidences_map.items():
            
            #track_pred_df = log_id_feather[log_id]
            #category = track_pred_df[track_pred_df['track_uuid'] == track_uuid]['category'].unique()[0]

            if (np.sum(np.where(track_scores > threshold, 1, 0))/len(track_scores)) > 0.5:
                #print(f'{prompt_id}: {category}')
                output_data[prompt_id][log_id].append(track_uuid)

        output_file = output_json_path / input_json_path.stem / f'{prompt_id}.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data[prompt_id], f, indent=4)

    return output_data

def convert_similarity_scores_to_tracker(tracker, log_prompt_pairs_path):

    with open(log_prompt_pairs_path, 'rb') as file:
        lpp = json.load(file)

    split = get_log_split(list(lpp.keys())[0])
    plp = swap_keys_and_listed_values(lpp)

    log_id_to_tracker_feather = {}
    log_id_to_clip_feather = {}
    for prompt, log_ids in tqdm(list(plp.items())):
        with open(f'baselines/clip_track/similarity_scores/{tracker}_{split}/{prompt}.json', 'rb') as file:
            track_ids = json.load(file)

        for log_id in log_ids:
            if log_id not in log_id_to_tracker_feather:
                log_id_to_tracker_feather[log_id] = pd.read_feather(paths.TRACKER_PRED_DIR / tracker / split / log_id / 'sm_annotations.feather')

            tracker_df = log_id_to_tracker_feather[log_id]
            clip_df = tracker_df[tracker_df['track_uuid'].isin(track_ids[log_id])]
            clip_df['prompt'] = prompt

            if log_id not in log_id_to_clip_feather:
                log_id_to_clip_feather[log_id] = clip_df
            else:
                log_id_to_clip_feather[log_id] = pd.concat([log_id_to_clip_feather[log_id], clip_df], axis=0)

    
    for log_id, clip_df in log_id_to_clip_feather.items():
        clip_tracker_path = Path(paths.TRACKER_PRED_DIR / (tracker + '_clip') / split / log_id / 'sm_annotations.feather')
        clip_tracker_path_csv = Path(paths.TRACKER_PRED_DIR / (tracker + '_clip') / split / log_id / 'sm_annotations.csv')
        clip_tracker_path.parent.mkdir(parents=True, exist_ok=True)
        print(clip_tracker_path)
    
        clip_df.to_csv(clip_tracker_path_csv)
        clip_df.to_feather(clip_tracker_path)

if __name__ == "__main__":
    tracker = 'Valeo4Cast_Tracking'
    split = 'val'

    try:
        eval_similarity_scores(f'/home/crdavids/Trinity-Sync/refbot/baselines/clip_track/similarity_scores/{tracker}_{split}.json',
                    'baselines/clip_track/similarity_scores', majority_threshold=.9)
        print('Converting to tracker ...')
        convert_similarity_scores_to_tracker(tracker, f'/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/log_prompt_pairs_{split}.json')
    except: pass
