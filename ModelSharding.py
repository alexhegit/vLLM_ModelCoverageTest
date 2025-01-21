from accelerate.utils import calculate_maximum_sizes
from accelerate.commands.estimate import *
from argparse import ArgumentParser
import pandas as pd
import torch
import os
import math
def main():
    free, total = torch.cuda.mem_get_info("cuda:0")
    gpu_total_mem_in_gb = total/1024/1024/1024

    parser = ArgumentParser(description="")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--token", type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    csv_file = args.csv
    token = args.token
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        model_id = row['model_id']
        try:
            dummy_model=create_empty_model(model_id, "transformers",True, token)
            total_size, _ = calculate_maximum_sizes(dummy_model)
            model_total_mem_in_gb = total_size/1024/1024/1024/2
            kv_margin_in_gb = 20 # 20 GB
        except:
            print("not supported")
        tp = math.ceil(model_total_mem_in_gb / (gpu_total_mem_in_gb - kv_margin_in_gb))
        df.loc[index, 'gpu (GB)'] = gpu_total_mem_in_gb
        df.loc[index, 'model (GB)'] = model_total_mem_in_gb
        df.loc[index, 'tp recommend'] = int(tp)
        base_name, ext = os.path.splitext(csv_file)
        output_csv_file = f"{base_name}_tpshard{ext}"
        df.to_csv(output_csv_file, index=False)
        
if __name__ == "__main__":
    main()
