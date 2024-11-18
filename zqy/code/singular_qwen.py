import os
import subprocess

import pandas as pd
import polars as pl
import re
from time import time
import logging

# import kaggle_evaluation.aimo_2_inference_server
from transformers import AutoModelForCausalLM, AutoTokenizer

# DIR_REPO = '/mnt/kaggleMATH/'
# DIR_REPO = os.path.join(os.path.expanduser("~"), "learningFile", "kaggleMATH")
DIR_REPO = os.path.join("/notebooks", "learningFile", "kaggleMATH")
repeat = 2
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
cache_dir = os.path.join(DIR_REPO, "zqy", "temp")


class TransformerSolver:
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = 2048
        self.max_input_tokens = 1024
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )

    def solve(self, prompt, reasoning_type="CoT"):
        if reasoning_type == "CoT":
            messages = [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": prompt},
            ]
        elif reasoning_type == "TIR":
            messages = [
                {
                    "role": "system",
                    "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": prompt},
            ]
        elif reasoning_type == "SHORT":
            messages = [
                {
                    "role": "system",
                    "content": "Please provide a short answer, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": prompt},
            ]
        else:
            raise ValueError("Unsupported reasoning type")

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        for key, value in model_inputs.items():
            logging.info(f"{key}: {value.size()}")

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response


def pred2int(pred):
    matches = re.findall(r"\\boxed{([^}]+)}", pred)
    tag = False
    for val in matches[-1::-1]:
        try:
            ans = int(val) % 1000
            tag = True
            break
        except:
            pass
    if tag:
        return ans
    else:
        return -1


def predict(data, output):
    for prob_i in range(data.shape[0]):
        question = data.loc[prob_i, "problem"]
        for i in range(repeat):
            logging.info("=====================================")
            logging.info(f"Problem id: {prob_i}")
            logging.info(f"repeat times: {i+1}")
            time1 = time()
            response = solver.solve(question, reasoning_type="CoT")
            # trim the prediction
            ans = pred2int(response)
            # check if the answer is valid
            # if not, try SHORT prompt
            if ans == -1:
                response = solver.solve(question, reasoning_type="SHORT")
                ans = pred2int(response)
            output.loc[prob_i, f"A{i+1}"] = ans
            output.to_csv(
                os.path.join(DIR_REPO, "zqy", "output", "output.csv"), index=False
            )
            time2 = time()
            logging.info(f"time expenditure: {time2 - time1}")
            logging.info(f"response: {response}")
            logging.info(f"answer: {ans}")
            logging.info("=====================================")


def create_file():
    os.makedirs(os.path.join(DIR_REPO, "zqy", "dataset"), exist_ok=True)
    os.makedirs(os.path.join(DIR_REPO, "zqy", "temp"), exist_ok=True)
    os.makedirs(os.path.join(DIR_REPO, "zqy", "output"), exist_ok=True)
    os.makedirs(os.path.join(DIR_REPO, "zqy", "log"), exist_ok=True)
    with open(os.path.join(DIR_REPO, "zqy", "log", "singular_qwen.log"), "w") as f:
        f.write(f"Log file created on {pd.Timestamp.now()}\n")


if __name__ == "__main__":
    create_file()
    logging.basicConfig(
        filename=os.path.join(DIR_REPO, "zqy", "log", "singular_qwen.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    # create a solver
    device = "cuda"  # the device to load the model onto
    solver = TransformerSolver(model_name, device)

    id_problem_path = os.path.join(DIR_REPO, "zqy", "dataset", "reference.csv")
    data = pd.read_csv(id_problem_path)
    output = data.copy()
    output.drop(columns=["problem"], inplace=True)
    for i in range(repeat):
        output[f"A{i+1}"] = None

    predict(data, output)
