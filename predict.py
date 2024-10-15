import os, sys
import pickle
from pathlib import Path
from termcolor import cprint
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig
from typing import Optional, List

from multiprocessing import Pool

from openai import OpenAI
import google.generativeai as genai
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from proc_bench.api_utils import sleep_and_retry
from proc_bench.constants import *


@sleep_and_retry
def predict(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    seed: int = 0,
    with_raw: bool = False
):
    if model in OPENAI_MODELS:
        client = OpenAI(api_key=os.environ["OPENAI_KEY"])
            
        messages = [] if system is None else [{"role": "system", "content": system}]
        messages += [{"role": "user", "content": prompt}]
        
        if "o1" in model:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=1.0,
                seed=seed
            )
        else:
            completion = client.chat.completions.create(
               model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=seed
            )

        if with_raw:
            return completion.choices[0].message.content, completion
        else:
            return completion.choices[0].message.content
        
    elif model in GOOGLE_MODELS:
        genai.configure(api_key=os.environ["GOOGLE_KEY"])
        client = genai.GenerativeModel(model, system_instruction=system)
        gen_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=max_tokens)
        
        response = client.generate_content(
            prompt, generation_config=gen_config, safety_settings=GOOGLE_SAFETY_SETTINGS
        )
        if with_raw:
            return response.text, response
        else:
            return response.text
                
    elif model in ANTHROPIC_MODELS:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"])

        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            system="" if system is None else system,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if with_raw:
            return response.content[0].text, response
        else:
            return response.content[0].text
    
    elif model in MISTRAL_MODELS:
        client = MistralClient(api_key=os.environ["MISTRAL_KEY"], timeout=360)
        
        messages = [] if system is None else [ChatMessage(role="system", content=system)]
        messages += [ChatMessage(role="user", content=prompt)]
        
        chat_response = client.chat(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            random_seed=seed,
        )

        if with_raw:
            return chat_response.choices[0].message.content, chat_response
        else:
            return chat_response.choices[0].message.content
    else:
        raise ValueError(f"Model {model} unknown.")

def predict_mp(mp_arg):
    save_dir, model, prompt_path, system, max_tokens, seed = mp_arg
    save_path = Path(os.path.join(save_dir, os.path.basename(prompt_path)))
    if not save_path.exists():
        prompt = open(prompt_path).read()
        pred, pred_raw = predict(model, prompt, system, max_tokens, seed, with_raw=True)
        print(f"Saving prediction to {save_path}.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pred)

        save_path_pickle = Path(str(save_path).replace("preds", "preds_raw").replace(".txt", ".pkl"))
        save_path_pickle.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_pickle, "wb") as f:
            pickle.dump(pred_raw, f)
    else:
        print(f"Skipping {save_path} as it already exists.")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    if args.models is None:
        cprint(f"Running prediction for all models (starting at {args.resume_model}).", "cyan")
        models = ALL_MODELS[args.resume_model:]
    else:
        cprint(f"Running prediction for models {args.models}.", "cyan")
        models = args.models
        
    if args.task_names is not None:
        cprint(f"Running prediction for tasks {args.task_names}.", "cyan")
        tasks = args.task_names
    else:
        cprint("Running prediction for all generated tasks.", "cyan")
        tasks = [os.path.basename(t) for t in natsorted(glob(f"experiment/{args.exp_name}/dataset/*"))]
    
    mp_args = []
    for i, model in enumerate(models):
        cprint(f"[{i + 1}/{len(models)}]: {model}.", "green")
        
        for task in tasks:
            prompts_dir = os.path.join("experiment", args.exp_name, "dataset", task, "prompt")
            save_dir = os.path.join("experiment", args.exp_name, "preds", model, task)
            os.makedirs(save_dir, exist_ok=True)
            
            if args.prompt_idxs is None:
                prompt_paths = natsorted(glob(os.path.join(prompts_dir, "*.txt")))
            else:
                prompt_paths = [os.path.join(prompts_dir, f"{str(prompt_idx).zfill(4)}.txt") for prompt_idx in args.prompt_idxs]
            
            for prompt_path in prompt_paths:
                mp_args.append((save_dir, model, prompt_path, args.system, args.max_tokens, args.seed))
    
    # with Pool(args.num_workers) as p:
    #     p.map(predict_mp, mp_args)
    
    for arg in mp_args:
        predict_mp(arg)

    # with Pool(args.num_workers) as p:
    #     for _ in tqdm(p.imap_unordered(predict_mp, mp_args), total=len(mp_args)):
    #         pass

if __name__ == '__main__':
    run()