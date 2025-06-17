import json

# Read JSON data from input file
input_file = "av2_sm_downloads/log_prompt_pairs_test.json"  # Replace with your input file path
with open(input_file, "r") as f:
    data = json.load(f)

# Extract unique entries
unique_entries = list(set(
    item for sublist in data.values() for item in sublist
))

# Save unique entries to output file
output_file = "results_tmp/log_prompt_pairs_test_unique.json"  # Output file path
with open(output_file, "w") as f:
    json.dump(unique_entries, f, indent=4)

print(f"Unique entries saved to {output_file}")