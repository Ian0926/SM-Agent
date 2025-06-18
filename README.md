
# SM-Agent: 1st Place Solution for Argoverse 2 Scenario Mining Challenge

This repository contains the official implementation for the 1st place solution in the CVPR 2025 WAD Argoverse 2 Scenario Mining Challenge.

## 1. Installation

First, create and activate a Conda environment.

```bash
conda create -n refav python=3.10
conda activate refav
```

Next, install the required libraries and packages.

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install other necessary packages
pip install tqdm pyarrow pandas scipy av2 pathos huggingface_hub pyvista transformers tabulate anthropic accelerate

# Install TrackEval
cd TrackEval
pip install .
cd ..
```

## 2. Data Preparation


### 2.1. Argoverse 2 Sensor Dataset

You need to download the Argoverse 2 Sensor dataset.

First, install `s5cmd`:
```bash
conda install s5cmd -c conda-forge
```

Then, run the command below to download the dataset. For more details, refer to the [Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data).

```bash
# Set your target directory
export TARGET_DIR="$HOME/data/av2"  # IMPORTANT: Change this to your desired location

# Download the sensor dataset
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/*" $TARGET_DIR
```

### 2.2. Argoverse 2 Scenario Mining Add-on

Download the official scenario-mining add-on, which contains the natural language queries.

```bash
# Set your target directory for this download
export TARGET_DIR="$(pwd)/downloads/av2_sm_downloads"

# Download the add-on
s5cmd --no-sign-request cp "s3://argoverse/tasks/scenario_mining/*" $TARGET_DIR
```

### 2.3. Tracking Predictions

Our method operates on 3D tracking predictions. Following [RefAV](https://github.com/CainanD/RefAV/tree/main), you have two options:

1.  **(Run Your Own)**: Generate tracking results using a state-of-the-art method like [LT3D](https://github.com/neeharperi/LT3D).
2.  **(Download General Predictions)**: Download pre-existing tracking predictions from the AV2 Tracker Predictions on [Hugging Face](https://huggingface.co/datasets/CainanD/AV2_Tracker_Predictions/tree/main) or [Google Drive](https://drive.google.com/file/d/1X19D5pBBO56eb_kvPOePLLhHDCsY0yql/view).

## 3. Configuration

Before running the code, you must configure the paths to the downloaded datasets.

Edit the file `refAV/paths.py` and update the following variables to point to the correct locations on your machine:

*   `AV2_DATA_DIR`: The path to the Argoverse 2 Sensor Dataset you downloaded in step 2.1.
*   `TRACKER_DOWNLOAD_DIR`: The path to the folder containing the tracking predictions (from step 2.3).
*   `SM_DOWNLOAD_DIR`: The path to the scenario-mining add-on (from step 2.2).

## 4. Running the Experiment

The workflow consists of generating code from descriptions and then executing that code to filter scenarios.

### Step 1: (Optional) Generate Code with Global Context

This approach processes all scenario descriptions at once to improve consistency.

**A. Get all unique prompts:**
```bash
python tools/get_unique_prompt.py
```

**B. Convert all unique descriptions into code:**
This step uses an LLM to generate the code. The example below uses a Gemini model name.
```bash
python run/language2code.py --exp_name exp_gemini-2.5-pro-preview-06-05
```

**C. Split the generated JSON into individual code files:**
```bash
python tools/json2file.py
```

### Step 2: Run Filtering and Auto-Correction

This is the main execution script. It uses the generated code to filter scenarios from the dataset.

**Key Feature**: If a script fails to execute or times out (default: 5 minutes), the system will automatically attempt to regenerate a corrected version of the code and re-run it.

```bash
python run/run_experiment.py --exp_name exp_gemini-2.5-pro-preview-06-05
```

After the run is complete, the results will be saved in the experiment's output directory.

## Acknowledgements

Our work is built upon the foundational framework provided by the [RefAV](https://github.com/CainanD/RefAV/tree/main) repository. We extend our sincere gratitude to the authors for making their code public.
