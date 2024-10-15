#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shutil import copy, copy2
import hashlib
import json
import pandas as pd
from pathlib import Path
import hydra

def calculate_md5(file_path):
    """ calculate md5 hash of a file """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def dir2df(in_dir):
    x_paths = sorted(in_dir.glob('**/prompt/*'))
    y_paths = sorted(in_dir.glob('**/label/*'))
    c_paths = sorted(in_dir.glob('**/config/*'))

    assert len(x_paths) == len(y_paths)
    assert len(x_paths) == len(c_paths)

    data = []
    for xp, yp, cp in zip(x_paths, y_paths, c_paths):
        task_name = xp.parent.parent.name
        example_name = xp.stem
        problem_name = task_name + '_' + example_name
        prompt = xp.read_text()
        with open(yp, 'r') as f:
            label = json.load(f)
        with open(cp, 'r') as f:
            config = json.load(f)
        line = {'prompt':prompt,
                'label':label,
                # 'config':config,
                'task_name':task_name,
                'example_name':example_name,
                'problem_name':problem_name}
        data.append(line)
    return pd.DataFrame(data)

def dir2parquet(in_dir, save_dir):
    df = dir2df(in_dir)
    tasknames = sorted(df.task_name.unique())
    for tn in tasknames:
        savepath = save_dir / (tn + '.parquet')
        savepath.parent.mkdir(parents=True, exist_ok=True)
        pdf = df.query('task_name == @tn')
        pdf.to_parquet(savepath, index=False)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args):
    prefix_dir = Path(f'./experiment/{args.exp_name}')
    save_dir = Path('./experiment/')

    gt_dir = prefix_dir / 'dataset'
    gtpaths = sorted(gt_dir.glob('**/label/**/*.json'))
    data = []
    for gp in gtpaths:
        example_id = gp.stem
        task_name = gp.parent.parent.name
        problem_name = task_name + '/' + example_id

        with open(gp, 'r') as f:
            y_true = json.load(f)
        
        len_interm = len(y_true['intermediate'])
        states = [y_true['init']] + y_true['intermediate'] + [y_true['final']]
        
        data.append([gp, task_name, problem_name, example_id,
                    y_true, y_true['init'], y_true['intermediate'], y_true['final'],
                    states, len_interm])

    gt_df = pd.DataFrame(data)
    gt_df.columns = ['path_gt', 'task_name', 'problem_name', 'example_id',
                    'y_true', 'y_init', 'y_inter', 'y_final', 'states', 'len_inter']
    gt_df['cnt'] = 1
    gt_df['hash_md5'] = [calculate_md5(row.path_gt) for _, row in gt_df.iterrows()]

    assert len(gt_df) == len(gt_df.hash_md5.unique())
    # hs = gt_df[gt_df.hash_md5.duplicated()].hash_md5.values[0]
    # gt_df[gt_df.hash_md5 == hs]

    gt_df['len_category'] = pd.cut(
        gt_df['len_inter'], 
        bins=[0, 5, 15, 24],  # len_inter: 1-5, 6-15, 16-24
                              # step :     2-6, 7-16, 17-25
        labels=['short', 'medium', 'long'],  # short: 1-5, medium: 6-15, long: 16-24
        right=True 
    )

    # create dummy variables for the category and add them to the original dataframe
    dummies = pd.get_dummies(gt_df['len_category'], prefix='length')
    gt_df = pd.concat([gt_df, dummies], axis=1)

    for _, row in gt_df.iterrows():
        path_dst = save_dir / 'level' / row.len_category / 'dataset' / row.task_name / 'label' / row.path_gt.name
        path_dst.parent.mkdir(exist_ok=True, parents=True)
        copy2(row.path_gt, path_dst)
        print('copy', row.path_gt, '\n -->', path_dst)
        
        path_src = str(row.path_gt).replace('/label/', '/config/')
        path_dst = Path(str(path_dst).replace('/label/', '/config/'))
        path_dst.parent.mkdir(exist_ok=True, parents=True)
        copy2(path_src, path_dst)
        print('copy', path_src, '\n -->', path_dst)

        path_src = str(row.path_gt).replace('/label/', '/prompt/').replace('.json', '.txt')
        path_dst = Path(str(path_dst).replace('/config/', '/prompt/').replace('.json', '.txt'))
        path_dst.parent.mkdir(exist_ok=True, parents=True)
        copy2(path_src, path_dst)
        print('copy', path_src, '\n -->', path_dst)
    
    savepath = prefix_dir / 'misc' / 'gt_df.csv'
    savepath.parent.mkdir(exist_ok=True, parents=True)
    gt_df.to_csv(savepath, index=False)
    pd.to_pickle(gt_df, savepath.with_suffix('.pkl'))

    prefix_dir = Path(f'./experiment/{args.exp_name}/dataset/')
    save_dir = Path('../experiment/{args.exp_name}/misc/dataset/')
    dir2parquet(prefix_dir, save_dir)

if __name__ == '__main__':
    main()