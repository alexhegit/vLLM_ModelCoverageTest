import os
import shutil
from datetime import datetime
import logging
from argparse import ArgumentParser
import pandas as pd
from vllm import LLM, SamplingParams
from multiprocessing import Process, Queue


def setup_logging():
    current_date = datetime.now().strftime('%Y%m%d')
    log_file = f"mct-{current_date}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def infer_with_model(model_id, gpus, result_queue):
    try:
        if not isinstance(gpus, int) or gpus <= 0:
            logging.error(f"<vLLM-CMT> Provided GPU count is invalid for model {model_id}: {gpus}")
            result_queue.put("FAILED")
            return

        tp = gpus
        logging.info(f"<vLLM-CMT> Inference Model {model_id}, TP {tp}")
        llm = LLM(
            model=model_id,
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
            raise ValueError("<vLLM-CMT> No outputs received from the model.")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        logging.info(f"<vLLM-CMT> Model {model_id} inference status: PASS")
        result_queue.put("PASS")
    except Exception as e:
        logging.error(f"<vLLM-CMT> Error during inference for model {model_id}: {e}")
        result_queue.put("FAILED")


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


def load_csv_file(csv_file):
    if not os.path.isfile(csv_file):
        logging.error(f"<vLLM-CMT> Provided CSV file does not exist: {csv_file}")
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")

    df = pd.read_csv(csv_file)
    if 'model_id' not in df.columns or 'gpus' not in df.columns:
        logging.error("<vLLM-CMT> CSV file must contain 'model_id' and 'gpus' columns.")
        raise ValueError("CSV file must contain 'model_id' and 'gpus' columns.")
    return df


def save_results_to_csv(df, csv_file):
    base_name, ext = os.path.splitext(csv_file)
    output_csv_file = f"{base_name}_results{ext}"
    df.to_csv(output_csv_file, index=False)
    logging.info(f"<vLLM-CMT> Results saved to: {output_csv_file}")


def main():
    setup_logging()
    parser = ArgumentParser(description="Run inference on models specified in a CSV file.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    try:
        df = load_csv_file(args.csv)
        results = []

        for index, row in df.iterrows():
            model_id = row['model_id']
            gpus = int(row['gpus'])

            result_queue = Queue()

            p = Process(target=infer_with_model, args=(model_id, gpus, result_queue))
            p.start()
            p.join()

            status = result_queue.get()
            results.append((model_id, gpus, status))

            df.loc[index, 'status'] = status
            save_results_to_csv(df, args.csv)

            delete_model_cache()

    except Exception as e:
        logging.error(f"<vLLM-CMT> Error occurred while processing CSV file or models: {e}")


if __name__ == "__main__":
    main()
