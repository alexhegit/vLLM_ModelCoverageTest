import os
import shutil
from datetime import datetime
import torch
import logging
from argparse import ArgumentParser
import pandas as pd
from vllm import LLM, SamplingParams

current_date = datetime.now().strftime('%Y%m%d')
log_file = f"mct-{current_date}.log"

logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class InferenceEngine:
    def __init__(self):
        pass
    
    def infer_with_model(self, model_id, gpus):
        logging.info(f"Starting inference for model: {model_id}, TP: {gpus}")
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
            tp = len(gpus)
            llm = LLM(model=model_id, tensor_parallel_size=tp, trust_remote_code=True, gpu_memory_utilization=0.95)
            prompts = ["The capital of France is"]
            outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.9))
            if not outputs or len(outputs) == 0:
                raise ValueError("No outputs received from the model.")
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            return "PASS"
        except Exception as e:
            logging.error(f"Error occurred while inferring model {model_id}: {e}")
            return "FAILED"

def delete_model_cache():
    cache_dir = os.path.expanduser("/root/.cache/huggingface/hub/")
    
    if os.path.exists(cache_dir):
        try:
            # Remove the entire cache directory
            shutil.rmtree(cache_dir)
            logging.info(f"Deleted cached models at: {cache_dir}")
        except Exception as e:
            logging.error(f"Error deleting cache at {cache_dir}: {e}")


def main():
    parser = ArgumentParser(description="Test models with specified prompts from a CSV file.")
    parser.add_argument("--csv", type=str, default="ml.csv",
                        help="Path to the CSV file containing model_id and gpus")
    args = parser.parse_args()
    
    if not args.csv:
        parser.print_help()
        return

    csv_file = args.csv
    engine = InferenceEngine()

    try:
        df = pd.read_csv(csv_file)
        if 'model_id' not in df.columns or 'gpus' not in df.columns:
            raise ValueError("CSV file must contain columns 'model_id' and 'gpus'")
        
        statuses = []
        for index, row in df.iterrows():
            model_id = row['model_id']
            gpus_str = str(row['gpus']).strip()
            try:
                gpus = list(map(int, gpus_str.split(',')))
            except ValueError as e:
                raise ValueError(f"Invalid gpus value '{gpus_str}' for model: {model_id}") from e
            
            status = engine.infer_with_model(model_id, gpus)
            statuses.append(status)
            delete_model_cache()
        
        # Add status column to the original DataFrame
        df['status'] = statuses
        df.to_csv(csv_file, index=False)

        print(f"Results saved to {csv_file} with 'status' column added.")

    except Exception as e:
        logging.error(f"Error occurred while reading CSV file or processing models: {e}")

if __name__ == "__main__":
    main()
