import os
import shutil
from datetime import datetime, timedelta
import torch
import logging
from argparse import ArgumentParser
import pandas as pd
from vllm import LLM, SamplingParams

def get_log_file_name(csv_file):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    current_date = datetime.now().strftime('%Y%m%d')
    return f"{base_name}_{current_date}.log"

class InferenceEngine:
    def __init__(self):
        pass
    
    def infer_with_model(self, model_id, gpus):
        start_time = datetime.now()
        logging.info("<vLLM-MCT> Starting inference for model: {model_id}, TP: {gpus}, Start time: {start_time}")
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
            tp = len(gpus)
            llm = LLM(model=model_id, tensor_parallel_size=tp, trust_remote_code=True, gpu_memory_utilization=0.95)
            prompts = ["The capital of France is"]
            outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.9))
            if not outputs or len(outputs) == 0:
                raise ValueError("<vLLM-MCT> No outputs received from the model.")
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                logging.info("<vLLM-MCT> Prompt: {prompt!r}, Generated text: {generated_text!r}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"<vLLM-MCT> PASS Inference completed for model: {model_id}, TP: {gpus}. Duration: {duration:.2f} seconds")
            return "PASS"
        except Exception as e:
            logging.error("<vLLM-MCT> Error occurred while inferring model {model_id}: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"<vLLM-MCT> FAILED Inference completed for model: {model_id}, TP: {gpus}. Duration: {duration:.2f} seconds")
            return "FAILED"

def delete_model_cache():
    cache_dir = os.path.expanduser("/root/.cache/huggingface/hub/")
    if os.path.exists(cache_dir):
        try:
            # Remove the entire cache directory
            shutil.rmtree(cache_dir)
            logging.info("<vLLM-MCT> Deleted cached models at: {cache_dir}")
        except Exception as e:
            logging.error(f"<vLLM-MCT> Error deleting cache at {cache_dir}: {e}")

def main():
    parser = ArgumentParser(description="Test models with specified prompts from a CSV file.")
    parser.add_argument("--csv", type=str, default="ml.csv",
                        help="Path to the CSV file containing model_id and gpus")
    args = parser.parse_args()
    
    if not args.csv:
        parser.print_help()
        return
    
    csv_file = args.csv
    log_file = get_log_file_name(csv_file)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    engine = InferenceEngine()
    
    try:
        df = pd.read_csv(csv_file)
        
        if 'model_id' not in df.columns or 'gpus' not in df.columns:
            raise ValueError("<vLLM-MCT> CSV file must contain columns 'model_id' and 'gpus'")
        
        statuses = []
        
        for index, row in df.iterrows():
            model_id = row['model_id']
            gpus_str = str(row['gpus']).strip()
            
            try:
                gpus = list(map(int, gpus_str.split(',')))
            except ValueError as e:
                raise ValueError(f"<vLLM-MCT> Invalid gpus value '{gpus_str}' for model: {model_id}") from e
            
            status = engine.infer_with_model(model_id, gpus)
            statuses.append(status)
            
            delete_model_cache()
        
        # Add status column to the original DataFrame
        df['status'] = statuses
        
        df.to_csv(csv_file, index=False)
    
    except Exception as e:
        logging.error(f"<vLLM-MCT> An error occurred: {e}")

if __name__ == "__main__":
    main()
