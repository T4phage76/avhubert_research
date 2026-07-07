import os, re
import pandas as pd
# Path to your txt file and your folder
txt_path = "visual_benefit/task_4db.txt"
folder = "visual_benefit/wordvariability_-4dB_AnAcVAnVAcV"

# Read the file and collect all filenames
all_files = []
with open(txt_path, "r") as f:
    for line in f:
        # Split by comma, strip whitespace, remove empty and 'NULL'
        names = [name.strip() for name in line.split(",") if name.strip() and name.strip() != "NULL"]
        all_files.extend(names)

unique_files = list(set(all_files))


# separate Ac, AcV, V, An and AnV

groups = {"An": [], "AnV": [], "Ac": [], "AcV": [], "V": []}

for fname in unique_files:
    # Determine group based on suffix pattern
    if "_a_-4dB" in fname:
        groups["An"].append(fname)
    elif re.search(r"_-4dB\.mp4$", fname):
        groups["AnV"].append(fname)
    elif "_v" in fname:
        groups["V"].append(fname)
    elif re.search(r"_a\.mp4$", fname):
        groups["Ac"].append(fname)
    else:
        groups["AcV"].append(fname)

pd.DataFrame(groups).to_csv("visual_benefit/stimuli_groups.csv", index=False)

# check data
df = pd.read_csv("visual_benefit/stimuli_groups.csv")