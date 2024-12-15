import pandas as pd
import json
from pathlib import Path
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Convert a CSV file to JSON format.")
parser.add_argument("source", type=str, help="Path to the source CSV file.")
parser.add_argument("--output", type=str, help="Path to the output JSON file. Defaults to <source stem>.json.")

args = parser.parse_args()

# Resolve input and output file paths
source_path = Path(args.source).resolve()
if not source_path.exists():
    raise FileNotFoundError(f"Source file not found: {source_path}")

output_path = Path(args.output).resolve() if args.output else source_path.with_suffix(".json")

# Load the CSV file
df = pd.read_csv(source_path)

# Convert DataFrame to JSON
json_output = df.to_json(orient="records", indent=4)

# Write JSON to the output file
with open(output_path, "w") as f:
    f.write(json_output)

print(f"JSON data has been written to {output_path}")
