import sys
import os
import re
import pandas as pd

def combine(path):
    parent = path
    permanova_frames = []
    anova_frames = []
    
    for name in os.listdir(parent):
        if not name.startswith("across_k_"):
            continue
        k_path = os.path.join(parent, name)
        if not os.path.isdir(k_path):
            continue
        
        # extract K from folder name
        m = re.search(r"across_k_(\d+)", name)
        if not m:
            continue
        K = int(m.group(1))
        
        # Look for run_ folders inside each across_k_ folder
        for run_name in os.listdir(k_path):
            if not run_name.startswith("run_"):
                continue
            run_path = os.path.join(k_path, run_name)
            if not os.path.isdir(run_path):
                continue
            
            # extract run number from folder name
            run_m = re.search(r"run_(\d+)", run_name)
            if not run_m:
                continue
            run_num = int(run_m.group(1))
            
            # permanova_summary.tsv
            perm_path = os.path.join(run_path, "permanova_summary.tsv")
            if os.path.exists(perm_path):
                df = pd.read_csv(perm_path, sep="\t")
                df["K"] = K
                df["run"] = run_num
                permanova_frames.append(df)
            
            # anova_results.csv
            # anova_path = os.path.join(run_path, "anova_results.csv")
            # if os.path.exists(anova_path):
            #     df = pd.read_csv(anova_path)
            #     df["K"] = K
            #     df["run"] = run_num
            #     anova_frames.append(df)
    
    # bind all together
    if permanova_frames:
        perm_all = pd.concat(permanova_frames, ignore_index=True)
        perm_all.to_csv("/tmp/permanova_summary_all.tsv", sep="\t", index=False)
    
    if anova_frames:
        anova_all = pd.concat(anova_frames, ignore_index=True)
        anova_all.to_csv("/tmp/anova_results_all.csv", index=False)
    
    print(f"Done. Processed {len(permanova_frames)} PERMANOVA files and {len(anova_frames)} ANOVA files.")
    print("Wrote permanova_summary_all.tsv and anova_results_all.csv")