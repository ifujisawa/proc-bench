import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Verdana'
from multiprocessing import Pool

from extract import create_format_df
from proc_bench.evaluator import exact_match, prefix_match_length

disp_model_names = {
    'claude-3-5-sonnet-20240620': 'Claude-3.5-Sonnet',
    'gemini-1.5-pro-latest': 'Gemini-1.5-Pro',
    'gpt-4o-2024-08-06': 'GPT-4o',
    'gpt-4o-mini-2024-07-18': 'GPT-4o-mini',
    'o1-mini-2024-09-12': 'o1-mini',
    'o1-preview-2024-09-12': 'o1-preview',
    'mistral-large-2407': 'Mistral-Large'
}

disp_task_names = {
    'task01': 'Sort',
    'task02': 'Gather',
    'task03': 'Count',
    'task04': 'Search',
    'task05': 'Copy',
    'task06': 'Substitute',
    'task07': 'Encode',
    'task08': 'Split1',
    'task09': 'Split2',
    'task10': 'Compose',
    'task11': 'Decompose',
    'task12': 'Rhythm',
    'task13': 'Compare',
    'task14': 'Count2',
    'task15': 'Decode',
    'task16': 'PushPop',
    'task17': 'Rotate',
    'task18': 'FillWord',
    'task19': 'DeleteChar',
    'task20': 'DeleteWord',
    'task21': 'Cumulate',
    'task22': 'MoveCyclic',
    'task23': 'FindCyclic'
}

def mp_eval(args):
    pp, gt_dir, predraw_dir, raw_dir = args
    example_id = pp.stem
    task_name = pp.parent.name
    problem_name = pp.parent.name + '/' + pp.stem
    model_name = pp.parent.parent.name

    with open(pp, 'r') as f:
        y_pred = json.load(f)
    
    gtpath = gt_dir / task_name / 'label' / (example_id + '.json')
    with open(gtpath, 'r') as f:
        y_true = json.load(f)
    predraw_path = predraw_dir / model_name / task_name / (example_id + '.txt')
    y_pred_raw = predraw_path.read_text()

    rawpath = raw_dir / model_name / task_name / (example_id + '.pkl')
    completion = pd.read_pickle(rawpath)

    if model_name == 'claude-3-5-sonnet-20240620':
        usage = completion.usage
        ct, pt = usage.output_tokens, usage.input_tokens
        tt, rt = ct + pt, 0
        fin_r = completion.stop_reason
    elif model_name == 'gemini-1.5-pro-latest':
        usage = completion.usage_metadata
        ct, pt = usage.candidates_token_count, usage.prompt_token_count
        tt, rt = usage.total_token_count, 0
        fin_r = str(completion.candidates[0].finish_reason)
    elif model_name == 'mistral-large-2407':
        usage = completion.usage
        ct, pt = usage.completion_tokens, usage.prompt_tokens
        tt, rt = usage.total_tokens, 0
        fin_r = str(completion.choices[0].finish_reason)
    else:
        usage = completion.usage
        ct, pt = usage.completion_tokens, usage.prompt_tokens
        tt, rt = usage.total_tokens, usage.completion_tokens_details.reasoning_tokens
        fin_r = completion.choices[0].finish_reason

    
    len_inter_true = len(y_true['intermediate'])
    len_inter_pred = len(y_pred['intermediate'])
    final_match = exact_match(y_true['final'], y_pred['final'])

    states_true = y_true['intermediate'] + [y_true['final']]
    states_pred = y_pred['intermediate'] + [y_pred['final']]

    len_step_true = len(states_true)
    len_step_pred = len(states_pred)

    pml = prefix_match_length(states_true, states_pred)
    denom = max(len_step_true, len_step_pred)
    prefix_accuracy = pml / denom
    sequential_match = float(prefix_accuracy == 1)
    zero_pml = (pml == 0.0)

    return [pp, gtpath, task_name, problem_name, model_name, example_id,
                y_true, y_pred_raw, y_pred, fin_r,
                ct, pt, tt, rt,
                len_inter_true, len_inter_pred, len_step_true, len_step_pred,
                final_match, pml, prefix_accuracy, sequential_match, zero_pml]


if __name__ == '__main__':
    prefix_dir = Path('./experiment/main/')

    gt_dir = prefix_dir / 'dataset'
    gtpaths = sorted(gt_dir.glob('**/label/**/*.json'))
    predraw_dir = prefix_dir / 'preds'
    pred_dir = prefix_dir / 'preds_json'
    predpaths = sorted(pred_dir.glob('**/*.json'))
    raw_dir = prefix_dir / 'preds_raw'
    rawpaths = sorted(raw_dir.glob('**/*.pkl'))

    format_df = create_format_df(prefix_dir)
    format_df.to_csv(prefix_dir / 'misc' / ('format_df.csv'), index=False, header=None)

    with Pool(os.cpu_count()) as p:
        data = p.map(mp_eval, [(pp, gt_dir, predraw_dir, raw_dir) for pp in predpaths])

    score_df = pd.DataFrame(data)
    score_df.columns = ['path_prediction', 'path_gt', 'task_name', 'problem_name', 'model_name', 'example_id',
                        'y_true', 'y_pred_raw', 'y_pred', 'finish_reason',
                        'tokens_completion', 'tokens_prompt', 'tokens_total', 'tokens_reasoning',
                        'len_inter_true', 'len_inter_pred', 'len_step_true', 'len_step_pred',
                        'score_fm', 'score_pml', 'score_pa', 'score_sm', 'score_zero_pml']
    score_df['cnt'] = 1
    score_df['no_pred'] = (score_df.y_pred_raw == '')
    task_names = score_df.task_name.unique()
    model_names = score_df.model_name.unique()
    score_df['len_category'] = pd.cut(
        score_df['len_inter_true'], 
        bins=[0, 5, 15, 24],                # len_inter: 1-5, 6-15, 16-24
                                            # step :     2-6, 7-16, 17-25
        labels=['short', 'medium', 'long'], # short: 1-5, medium: 6-15, long: 16-24
        right=True 
    )
    score_df['model_name'] = [disp_model_names[mn] for mn in score_df.model_name.values]
    score_df['task_name'] = [disp_task_names[tn] for tn in score_df.task_name.values]
    pd.to_pickle(score_df, prefix_dir / 'misc' / 'score_df.pkl')

    score_df = pd.read_pickle(prefix_dir / 'misc' / 'score_df.pkl')

    save_dir = prefix_dir / 'misc' / 'fig_table'
    save_dir.mkdir(parents=True, exist_ok=True)

    sdf = score_df.groupby(['model_name', 'len_category'], observed=False)[['score_sm', 'score_fm', 'score_pa']].mean().unstack()
    sdf2 = sdf.swaplevel(0, 1, axis=1).T.sort_index().T
    sdf2.to_csv(save_dir / 'table_ModelPerformance.csv')

    sdf3 = score_df.groupby(['model_name'], observed=False)[['score_sm', 'score_fm', 'score_pa']].mean()
    sdf3.to_csv(save_dir / 'table_ModelPerformance_overall.csv')

    stat_df = score_df.groupby(['task_name', 'model_name'])['score_sm'].mean().unstack()
    sort_model = 'o1-preview'  # Replace with the actual model name you want to sort by
    if sort_model not in stat_df.columns:
        raise ValueError(f"The model '{sort_model}' does not exist in the data.")
    stat_df = stat_df.sort_values(by=sort_model, ascending=False)
    n_tasks, n_models = stat_df.shape
    data = stat_df.values  # Shape: (n_tasks, n_models)
    task_names = stat_df.index.tolist()
    model_names = stat_df.columns.tolist()
    cmap = plt.get_cmap('tab20b')  # Updated to use plt.get_cmap()
    colors = [cmap(i) for i in range(n_models)]
    bar_width = 0.8 / n_models
    alpha = 0.8
    indices = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8, max(6, n_tasks * 0.5)), dpi=300)
    for i in range(n_models):
        ax.barh(indices + i * bar_width, data[:, i], bar_width,
                label=model_names[i], alpha=alpha, color=colors[i])
    ax.set_yticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_yticklabels(task_names)
    ax.set_ylabel('Tasks')
    ax.set_xlabel('Sequential Match')
    ax.set_title('Average Sequential Match Across Tasks and Models')
    ax.legend(fontsize=10)
    plt.grid(axis='x', color='lightgray', lw=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_SM.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    stat_df = score_df.groupby(['task_name', 'model_name'])['score_fm'].mean().unstack()
    # sort_model = 'o1-preview-2024-09-12'  # Replace with the actual model name you want to sort by
    # if sort_model not in stat_df.columns:
    #     raise ValueError(f"The model '{sort_model}' does not exist in the data.")
    stat_df = stat_df.sort_values(by=sort_model, ascending=False)
    n_tasks, n_models = stat_df.shape
    data = stat_df.values  # Shape: (n_tasks, n_models)
    # task_names = stat_df.index.tolist()
    model_names = stat_df.columns.tolist()
    cmap = plt.get_cmap('tab20b')  # Updated to use plt.get_cmap()
    colors = [cmap(i) for i in range(n_models)]
    bar_width = 0.8 / n_models
    alpha = 0.8
    indices = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8, max(6, n_tasks * 0.5)), dpi=300)
    for i in range(n_models):
        ax.barh(indices + i * bar_width, data[:, i], bar_width,
                label=model_names[i], alpha=alpha, color=colors[i])
    ax.set_yticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_yticklabels(task_names)
    ax.set_ylabel('Tasks')
    ax.set_xlabel('Final Match')
    ax.set_title('Average Final Match Across Tasks and Models')
    ax.legend(fontsize=10)
    plt.grid(axis='x', color='lightgray', lw=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_FM.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    stat_df = score_df.groupby(['task_name', 'model_name'])['score_pa'].mean().unstack()
    # sort_model = 'o1-preview-2024-09-12'  # Replace with the actual model name you want to sort by
    # if sort_model not in stat_df.columns:
    #     raise ValueError(f"The model '{sort_model}' does not exist in the data.")
    stat_df = stat_df.sort_values(by=sort_model, ascending=False)
    n_tasks, n_models = stat_df.shape
    data = stat_df.values  # Shape: (n_tasks, n_models)
    # task_names = stat_df.index.tolist()
    model_names = stat_df.columns.tolist()
    cmap = plt.get_cmap('tab20b')  # Updated to use plt.get_cmap()
    colors = [cmap(i) for i in range(n_models)]
    bar_width = 0.8 / n_models
    alpha = 0.8
    indices = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(8, max(6, n_tasks * 0.5)), dpi=300)
    for i in range(n_models):
        ax.barh(indices + i * bar_width, data[:, i], bar_width,
                label=model_names[i], alpha=alpha, color=colors[i])
    ax.set_yticks(indices + bar_width * (n_models - 1) / 2)
    ax.set_yticklabels(task_names)
    ax.set_ylabel('Tasks')
    ax.set_xlabel('Prefix Accuracy')
    ax.set_title('Average Prefix Accuracy Across Tasks and Models')
    ax.legend(fontsize=10)
    plt.grid(axis='x', color='lightgray', lw=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_PA.png', bbox_inches='tight')
    # plt.show()
    plt.close()


    def fig_model_step(score_df, score_name, score_name_title, save_dir):
        len_step_values = np.sort(score_df['len_step_true'].unique())
        plt.figure(figsize=(6, 4), dpi=300)
        for mn in model_names:
            pdf = score_df[score_df['model_name'] == mn]
            arr = []
            for leng in len_step_values:
                pdf2 = pdf[pdf['len_step_true'] == leng]
                mean_value = pdf2[score_name].mean()
                arr.append(mean_value)
            arr = np.array(arr)
            plt.plot(len_step_values, arr, marker='o', label=mn, alpha=.8, markersize=4)
        # Set plot limits and labels
        if score_name.find('pml') != -1:
            plt.scatter(len_step_values, len_step_values, marker='x')
        else:
            plt.ylim(0, 1.02)
        plt.grid(axis='y')
        plt.xlabel('Problem Length', fontsize=11)
        plt.title(score_name_title, fontsize=14)
        # plt.title('Mean Score vs. Length of Intermediate Steps for Multiple Models')
        # plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(save_dir / f'fig_model_step_{score_name}.png', bbox_inches='tight')
        # plt.show()
        plt.close()

    fig_model_step(score_df, 'score_pa', 'Prefix Accuracy', save_dir)
    fig_model_step(score_df, 'score_fm', 'Final Match', save_dir)
    fig_model_step(score_df, 'score_sm', 'Sequential Match', save_dir)
    fig_model_step(score_df, 'score_pml', 'Prefix Match Length', save_dir)

    # Create a legend figure
    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for mn in model_names:
        plt.plot([], [], marker='o', label=' ' + mn, alpha=.8, markersize=4)
    legend = ax.legend(frameon=False, handletextpad=0, ncol=4, columnspacing=1)
    legend_fig = legend.figure
    legend_fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
    fig.savefig(save_dir / 'fig_legend.png', bbox_inches=bbox, dpi=300)
    # plt.show()
    plt.close()

    plt.figure(figsize=(len(model_names)*4, len(task_names)*3), dpi=300)
    cnt = 1
    for tn in task_names:
        for mn in model_names:
            pdf = score_df.query('model_name == @mn and task_name == @tn')
            arr = []
            len_step_values = sorted(pdf.len_step_true.unique())
            for leng in len_step_values:
                pdf2 = pdf.query('len_step_true == @leng')
                arr.append(pdf2.score_pml.mean())
            arr = np.array(arr)
            plt.subplot(len(task_names), len(model_names), cnt)
            plt.bar(len_step_values, arr)
            plt.grid(axis='y')
            plt.scatter(len_step_values, len_step_values, marker='x')
            plt.xlabel('Problem Length', fontsize=7)
            plt.ylabel('Prefix Match Length', fontsize=7)
            plt.title(f'{mn}: {tn}', fontsize=8)
            cnt += 1

    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_PML.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    plt.figure(figsize=(len(model_names)*4, len(task_names)*3), dpi=200)
    cnt = 1
    for tn in task_names[:12]:
        for mn in model_names:
            pdf = score_df.query('model_name == @mn and task_name == @tn')
            arr = []
            len_step_values = sorted(pdf.len_step_true.unique())
            for leng in len_step_values:
                pdf2 = pdf.query('len_step_true == @leng')
                arr.append(pdf2.score_pml.mean())
            arr = np.array(arr)
            plt.subplot(len(task_names), len(model_names), cnt)
            plt.bar(len_step_values, arr)
            plt.grid(axis='y')
            plt.scatter(len_step_values, len_step_values, marker='x')
            plt.xlabel('Problem Length', fontsize=7)
            plt.ylabel('Prefix Match Length', fontsize=7)
            plt.title(f'{mn}: {tn}', fontsize=8)
            cnt += 1

    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_PML_part1.png', bbox_inches='tight')
    # plt.show()
    plt.close()    
    plt.figure(figsize=(len(model_names)*4, len(task_names)*3), dpi=200)
    cnt = 1
    for tn in task_names[12:]:
        for mn in model_names:
            pdf = score_df.query('model_name == @mn and task_name == @tn')
            arr = []
            len_step_values = sorted(pdf.len_step_true.unique())
            for leng in len_step_values:
                pdf2 = pdf.query('len_step_true == @leng')
                arr.append(pdf2.score_pml.mean())
            arr = np.array(arr)
            plt.subplot(len(task_names), len(model_names), cnt)
            plt.bar(len_step_values, arr)
            plt.grid(axis='y')
            plt.scatter(len_step_values, len_step_values, marker='x')
            plt.xlabel('Problem Length', fontsize=7)
            plt.ylabel('Prefix Match Length', fontsize=7)
            plt.title(f'{mn}: {tn}', fontsize=8)
            cnt += 1

    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_PML_part2.png', bbox_inches='tight')
    # plt.show()
    plt.close()    

    plt.figure(figsize=(len(model_names)*4, len(task_names)*3), dpi=300)
    cnt = 1
    for tn in ['FindCyclic', 'Compare', 'Sort']:
        for mn in model_names:
            pdf = score_df.query('model_name == @mn and task_name == @tn')
            arr = []
            len_step_values = sorted(pdf.len_step_true.unique())
            for leng in len_step_values:
                pdf2 = pdf.query('len_step_true == @leng')
                arr.append(pdf2.score_pml.mean())
            arr = np.array(arr)
            plt.subplot(len(task_names), len(model_names), cnt)
            plt.bar(len_step_values, arr)
            plt.grid(axis='y')
            plt.scatter(len_step_values, len_step_values, marker='x')
            plt.xlabel('Problem Length', fontsize=12)
            plt.ylabel('Prefix Match Length', fontsize=12)
            plt.title(f'{mn}: {tn}', fontsize=12)
            cnt += 1

    plt.tight_layout()
    plt.savefig(save_dir / 'fig_model_task_PML_selected.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    for mn in model_names:
        pdf = score_df.query('model_name == @mn')
        plt.scatter(pdf.len_inter_true.values, pdf.len_inter_pred.values, alpha=.1, edgecolor=None, s=15)
        plt.plot(np.arange(26), np.arange(26), color='C1', alpha=.4)
        plt.title(mn)
        plt.xlabel('Problem Length')
        plt.ylabel('Prediction Length')
        plt.ylim(-5, 50)
        plt.savefig(save_dir / f'fig_model_length_gt_pred_{mn}.png', bbox_inches='tight')
        # plt.show()
        plt.close()


    ### Distribution of Model Scores Across Tasks
    sdf = score_df.groupby(['model_name', 'task_name'], observed=False)[['score_sm']].mean().unstack().T
    sdf['median'] = sdf.median(axis=1)
    sdf.sort_values('median', ascending=False, inplace=True)
    temp_task_names = sdf.reset_index().task_name.values

    plt.figure(figsize=(10, 2.3), dpi=300)
    plt.boxplot(np.array(sdf).T)
    plt.xticks(range(1, 24), [])
    plt.title("Sequential Match", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_task_model_SM.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    sdf = score_df.groupby(['model_name', 'task_name'], observed=False)[['score_pa']].mean().unstack().T
    sdf = sdf.reset_index().set_index('task_name').iloc[:, 1:]
    sdf = sdf.loc[temp_task_names]

    plt.figure(figsize=(10, 2.3), dpi=300)
    plt.boxplot(np.array(sdf).T)
    plt.xticks(range(1, 24), [])
    plt.title("Prefix Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_task_model_PA.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    sdf = score_df.groupby(['model_name', 'task_name'], observed=False)[['score_fm']].mean().unstack().T
    sdf = sdf.reset_index().set_index('task_name').iloc[:, 1:]
    sdf = sdf.loc[temp_task_names]

    plt.figure(figsize=(10, 3), dpi=300)
    plt.boxplot(np.array(sdf).T)
    plt.xticks(range(1, 24), sdf.reset_index().task_name.values, rotation=45)
    plt.title("Final Match", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_task_model_FM.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    sdf = score_df.groupby(['model_name', 'task_name'], observed=False)[['score_pa']].mean().unstack().T
    sdf['median'] = sdf.median(axis=1)
    sdf.sort_values('median', ascending=False, inplace=True)
    temp_task_names = sdf.reset_index().task_name.values
    # sdf = sdf.loc[temp_task_names]

    plt.figure(figsize=(10, 3), dpi=300)
    plt.boxplot(np.array(sdf).T)
    plt.xticks(range(1, 24), sdf.reset_index().task_name.values, rotation=45)
    plt.title("Prefix Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'fig_task_model_PA_xtick.png', bbox_inches='tight')
    # plt.show()
    plt.close()


    # Define success and failure conditions
    def pa_bin(row):
        if row['score_pa'] == 0:
            return 'PA = 0'
        elif row['score_pa'] == 1:
            return 'PA = 1'
        elif (0 < row['score_pa']) and (row['score_pa'] <= 0.5):
            return '0 < PA <= 0.5'
        else:
            return '0.5 < PA < 1'

    for mn in model_names:
        pdf = score_df.query('model_name == @mn').copy()
        pdf['pa_bin'] = pdf.apply(pa_bin, axis=1)
        grouped = pdf.groupby(['len_step_true', 'pa_bin']).size().unstack(fill_value=0)
        total_counts = grouped.sum(axis=1)
        proportions = grouped.div(total_counts, axis=0)

        plt.figure(figsize=(6, 4), dpi=300)
        data = proportions[['PA = 1', '0.5 < PA < 1', '0 < PA <= 0.5', 'PA = 0']]
        
        # インデックス
        ind = np.arange(len(data))
        width = 0.7  # バーの幅
        
        # 各カテゴリの積み重ねたバーの位置を計算
        bar1 = data['PA = 1']
        bar2 = data['0.5 < PA < 1']
        bar3 = data['0 < PA <= 0.5']
        bar4 = data['PA = 0']
        
        # 積み重ねバーの描画
        plt.bar(ind, bar1, width, label='PA = 1', color='C0', alpha=0.75)
        plt.bar(ind, bar2, width, bottom=bar1, label='0.5 < PA < 1', color='C8', alpha=0.75)
        plt.bar(ind, bar3, width, bottom=bar1 + bar2, label='0 < PA <= 0.5', color='lightgray', alpha=0.75)
        plt.bar(ind, bar4, width, bottom=bar1 + bar2 + bar3, label='PA = 0', color='darkgray', alpha=0.75)
        
        # グラフの調整
        plt.xlabel('Problem Length')
        plt.ylabel('Proportion')
        plt.title(mn)
        plt.xticks(ind, data.index)  # インデックスに基づくx軸のラベル
        plt.grid(axis='y')
        plt.legend(framealpha=1.0, loc='lower left')
        
        # グラフの表示
        plt.tight_layout()
        plt.savefig(save_dir / f'fig_model_step_PAbin_{mn}.png', bbox_inches='tight')
        # plt.show()
        plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    for mn in model_names:
        data = []
        pdf = score_df.query('model_name == @mn')
        for pml_lower in range(26):
            pdf2 = pdf[(pdf.len_step_true >= pml_lower)].copy()
            num_target = len(pdf2)
            num_higher = (pdf2.score_pml >= pml_lower).sum()
            data.append([num_target, num_higher])
        sdf = pd.DataFrame(data)
        sdf[2] = sdf.iloc[:, 1] / sdf.iloc[:, 0]
        plt.plot(sdf.iloc[:, 2].values, label=mn, marker='o', markersize=4, alpha=.7)
    plt.legend(loc='upper right', fontsize=9)
    plt.xlabel('Step Threshold', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.title('Proportion of Correct Predictions at or above Threshold', fontsize=12)
    plt.savefig(save_dir / 'fig_model_step_threshold.png', bbox_inches='tight')
    # plt.show()
    plt.close()