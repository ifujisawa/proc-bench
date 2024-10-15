import os
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

from pydantic import BaseModel
from typing import List, Any, Type
import hydra
from omegaconf import DictConfig

from openai import OpenAI

from proc_bench.api_utils import sleep_and_retry

def create_format_df(prefix_dir, tasktype='task'):
    prefix_dir = Path(prefix_dir)
    gt_dir = prefix_dir / 'dataset'
    save_dir = prefix_dir / 'misc'

    gtpaths = sorted(Path(gt_dir).glob('**/label/**/*.json'))
    tasknames = np.unique([gp.parent.parent.stem for gp in gtpaths])
    task_dirs = [gtpaths[0].parent.parent.parent / taskname for taskname in tasknames]
    format_gtpaths = [sorted(td.glob('**/label/**/*.json'))[-1] for td in task_dirs]

    data = []
    for tn, gp in zip(tasknames, format_gtpaths):
        with open(gp, 'r') as f:
            gt = json.load(f)

        if tasktype == 'task':
            if type(gt['final']) == list:
                ft1 = type(gt['final'][0]).__name__
            else:
                ft1 = None
            if type(gt['intermediate'][0]) == list:
                it2 = type(gt['intermediate'][0][0]).__name__
            else:
                it2 = None
            data.append([type(gt['intermediate'][0]).__name__, it2,
                        type(gt['final']).__name__, ft1,
                        gt['intermediate'][0], gt['final']])
            format_df = pd.DataFrame(data)
            format_df.columns = ['interm', 'interm_nest', 'final', 'final_nest', 'example_interm', 'example_final']
        elif tasktype == 'subtask':
            if type(gt['answer']) == list:
                ft1 = type(gt['answer'][0]).__name__
            else:
                ft1 = None
            data.append([type(gt['answer']).__name__, ft1, gt['answer']])
            format_df = pd.DataFrame(data)
            format_df.columns = ['answer', 'answer_nest', 'example_answer']
    format_df.index = tasknames
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(format_df, save_dir / 'format_df.pkl')
    return format_df

def create_result_df(prefix_dir, tasktype='task'):
    prefix_dir = Path(prefix_dir)
    data_dir = prefix_dir / 'dataset'
    pred_dir = prefix_dir / 'preds'
    save_dir = prefix_dir / 'misc'
    
    pred_paths = sorted(pred_dir.rglob('**/*.txt'))

    if tasktype == 'task':
        gt_paths, cfg_paths, x_paths = [], [], []
        for pp in pred_paths:
            gt_paths.append(data_dir / pp.parent.name / 'label' / (pp.stem + '.json'))
            cfg_paths.append(data_dir / pp.parent.name / 'config' / (pp.stem + '.json'))
            x_paths.append(data_dir / pp.parent.name / 'prompt' / (pp.stem + '.txt'))
        
        dt = []
        for gp, cp, xp, pp in zip(gt_paths, cfg_paths, x_paths, pred_paths):
            with open(xp, 'r') as f:
                x = f.read()
            with open(gp, 'r') as f:
                gt = json.load(f)
            with open(cp, 'r') as f:
                cfg = json.load(f)
            try:
                with open(pp, 'r', encoding='utf-8') as f:
                    pred = f.read()
            except:
                with open(pp, 'r', encoding='shift-jis') as f:
                    pred = f.read()
                print('shift-jis', pp)
            problem_name = gp.parent.parent.name + '/' + gp.stem
            dt.append([gp, cp, xp, pp, problem_name, x, gt, gt['final'], gt['init'], gt['intermediate'], pred, cfg])
        
        df = pd.DataFrame(dt)
        df.columns = ['path_gt', 'path_config', 'path_prompt', 'path_prediction', 'problem_name',
                    'prompt', 'gt_all', 'gt_final', 'gt_init', 'gt_intermediate', 'pred', 'config']
        result_df = df.copy()
        result_df['task_name'] = result_df.problem_name.str.split('/', expand=True).iloc[:, 0]
        result_df['model_name'] = [row.parent.parent.name for row in result_df.path_prediction.values]

        save_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(result_df, save_dir / 'result_df.pkl')
        return result_df
    elif tasktype == 'subtask':
        gt_paths, x_paths = [], []
        for pp in pred_paths:
            gt_paths.append(data_dir / pp.parent.name / 'label' / (pp.stem + '.json'))
            x_paths.append(data_dir / pp.parent.name / 'prompt' / (pp.stem + '.txt'))
        
        dt = []
        for gp, xp, pp in zip(gt_paths, x_paths, pred_paths):
            with open(xp, 'r') as f:
                x = f.read()
            with open(gp, 'r') as f:
                gt = json.load(f)
            try:
                with open(pp, 'r', encoding='utf-8') as f:
                    pred = f.read()
            except:
                with open(pp, 'r', encoding='shift-jis') as f:
                    pred = f.read()
                print('shift-jis', pp)
            problem_name = gp.parent.parent.name + '/' + gp.stem
            dt.append([gp, xp, pp, problem_name, x, gt, pred])
        
        df = pd.DataFrame(dt)
        df.columns = ['path_gt', 'path_prompt', 'path_prediction', 'problem_name',
                    'prompt', 'gt_all', 'pred']
        result_df = df.copy()
        result_df['task_name'] = result_df.problem_name.str.split('/', expand=True).iloc[:, 0]
        result_df['model_name'] = [row.parent.parent.name for row in result_df.path_prediction.values]

        save_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(result_df, save_dir / 'result_df.pkl')
        return result_df

def get_task_format(taskname, format_df, tasktype) -> Type[BaseModel]:
    types = format_df.loc[taskname]
    if tasktype == 'task':
        if eval(types['interm']) == list:
            format_interm = List[List[eval(types['interm_nest'])]]
        else:
            format_interm = List[eval(types['interm'])]
        if eval(types['final']) == list:
            format_final = List[eval(types['final_nest'])]
        else:
            format_final = eval(types['final'])

        # define the class for format dynamically
        class TaskFormat(BaseModel):
            intermediate: format_interm
            final: format_final
        return TaskFormat
    elif tasktype == 'subtask':
        if eval(types['answer']) == list:
            format_answer = List[eval(types['answer_nest'])]
        else:
            format_answer = eval(types['answer'])

        # define the class for format dynamically
        class TaskFormat(BaseModel):
            answer: format_answer
        return TaskFormat

@sleep_and_retry
def align_format(prompt, system_prompt, TaskFormat):
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=TaskFormat,
    )
    pred_raw = completion
    pred = pred_raw.choices[0].message.parsed
    return pred

def align_format_mp(mp_arg):
    row, format_df, save_dir, tasktype = mp_arg
    pred_path = row.path_prediction
    savepath = save_dir / row.model_name / row.task_name / (pred_path.stem + '.json')
    if tasktype == 'task':
        system_prompt = "You are an expert at structured data extraction. You will be given unstructured text and should convert it into the given structure. The task is consisting of a problem statement and a person's answer. The answer is in free-form text, but I would like to format it for evaluation purposes. Please convert the free-form text into the following JSON format. Do not include the final state in the last element of the intermediate list."
        vacant_pred = {'intermediate':[], 'final':''}
    elif tasktype == 'subtask':
        system_prompt = "You are an expert at structured data extraction. You will be given unstructured text and should convert it into the given structure. The task is consisting of a problem statement and a person's answer. The answer is in free-form text, but I would like to format it for evaluation purposes. Please convert the free-form text into the following JSON format. Extract only the final result, which can be a list or string. Given the list, you should format it as a list of strings."
        vacant_pred = {'answer':''}

    if savepath.exists():
        print(f'skipped. {savepath} already exists.')
    else:
        prompt = f'# Answer of a model:\n{row.pred}'
        TaskFormat = get_task_format(row.task_name, format_df, tasktype)
        try:
            pred = eval(align_format(prompt, system_prompt, TaskFormat).json())
        except:
            pred = vacant_pred
            print(f'Failed to format, empty prediction is inserted: {savepath}')

        savepath.parent.mkdir(parents=True, exist_ok=True)
        with open(savepath, 'w') as f:
            json.dump(pred, f, sort_keys=True)
        print(f'saved to {savepath}')

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    if args.exp_name in ['subtask', 'serial_subtask']:
        tasktype = 'subtask'
    else:
        tasktype = 'task'
    prefix_dir = Path(f'./experiment/{args.exp_name}')
    result_df = create_result_df(prefix_dir, tasktype=tasktype)
    format_df = create_format_df(prefix_dir, tasktype=tasktype)

    save_dir = prefix_dir / 'preds_json'
    rows = [row for _, row in result_df.iterrows()]
    mp_args = [(row, format_df, save_dir, tasktype) for row in rows]
    with Pool(args.num_workers) as p:
        p.map(align_format_mp, mp_args)

if __name__ == '__main__':
    run()