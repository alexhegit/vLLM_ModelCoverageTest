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
logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class InferenceEngine:
    def infer_with_model(self, model_id, gpus):
        try:
            if not isinstance(gpus, list) or len(gpus) == 0:
                logging.error(f"<vLLM-CMT> Provided GPUs list is invalid for model {model_id}: {gpus}")
                return "FAILED"
            
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            tp = len(gpus)
            logging.info(f"<vLLM-CMT> Inference Model {model_id}, TP {tp}")
            llm = LLM(model=model_id,
                      tensor_parallel_size=tp,
                      trust_remote_code=True,
                      gpu_memory_utilization=0.95,
                      max_model_len=1024,
                      enforce_eager=True,
                      load_format="dummy"
                     )
            prompts = ["The capital of France is"]
            outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.9))
            if not outputs or len(outputs) == 0:
                raise ValueError("<vLLM-CMT >No outtputs received from the model.")
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            logging.info(f"<vLLM-CMT> Model {model_id} inference status: PASS")
            return "PASS"
        except Exception as e:
            logging.error(f"<vLLM-CMT> Error during inference for model {model_id}: {e}")
            return "FAILED"

def delete_model_cache():
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logging.info(f"<vLLM-CMT> Model cache directory deleted: {cache_dir}")
        else:
            logging.warning(f"<vLLM-CMT> Model cache directory does not exist: {cache_dir}")
    except Exception as e:
        logging.error(f"<vLLM-CMT> Error occurred while deleting model cache: {e}")

def main():
    parser = ArgumentParser(description="Run inference on models specified in a CSV file.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    csv_file = args.csv
    if not os.path.isfile(csv_file):
        logging.error(f"<vLLM-CMT> Provided CSV file does not exist: {csv_file}")
        parser.print_help()
        return
    
    try:
        df = pd.read_csv(csv_file)
        if 'model_id' not in df.columns or 'gpus' not in df.columns:
            logging.error("<vLLM-CMT> CSV file must contain 'model_id' and 'gpus' columns.")
            raise ValueError("CSV file must contain 'model_id' and 'gpus' columns.")
        
        engine = InferenceEngine()
        results = []
        for index, row in df.iterrows():
            model_id = row['model_id']
            gpus = [int(gpu) for gpu in str(row['gpus']).split(',')]
            status = engine.infer_with_model(model_id, gpus)
            logging.info(f"<vLLM-CMT> Model {model_id} inference status: {status}")
            results.append(status)
            
            # Save intermediate results after each model
            df.loc[index, 'status'] = status
            base_name, ext = os.path.splitext(csv_file)
            output_csv_file = f"{base_name}_results{ext}"
            df.to_csv(output_csv_file, index=False)
            logging.info(f"<vLLM-CMT> Intermediate results saved to: {output_csv_file}")
            
            delete_model_cache()
        
    except Exception as e:
        logging.error(f"<vLLM-CMT> Error occurred while processing CSV file or models: {e}")

if __name__ == "__main__":
    main()
