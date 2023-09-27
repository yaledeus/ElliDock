import os
import sys
import json

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

result_dir = './results'

font_path = font_manager.findfont(font_manager.FontProperties(family="Times New Roman"))

font = {
    # 'family': font_manager.FontProperties(fname=font_path).get_name(),
    # 'weight': 'bold',
    'size': 11
}

def scatterplot_irmsd_crmsd(dataset, save_dir=os.getcwd()):
    plt.figure()
    plt.rc('font', **font)
    for file in os.listdir(result_dir):
        if not file.startswith(dataset.lower()):
            continue
        with open(os.path.join(result_dir, file), 'r') as fp:
            data = json.load(fp)
            sns.scatterplot(x='IRMSD', y='CRMSD', label=data['model_type'], data=data, legend=False)
    plt.xticks()
    plt.yticks()
    plt.xlabel('IRMSD', weight='bold', size=14)
    plt.ylabel('CRMSD', weight='bold', size=14)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{dataset}-scatter-crmsd-irmsd.png'))


def violinplot(dataset, metric, save_dir=os.getcwd()):
    plt.figure()
    plt.rc('font', **font)
    data = {
        'model_type': [],
        metric: []
    }
    for file in os.listdir(result_dir):
        if not file.startswith(dataset.lower()):
            continue
        with open(os.path.join(result_dir, file), 'r') as fp:
            item = json.load(fp)
            data['model_type'].extend([item['model_type'] for _ in range(len(item[metric]))])
            data[metric].extend(item[metric])
    sns.violinplot(x='model_type', y=metric, data=data, legend=False)
    plt.xticks()
    plt.yticks()
    plt.xlabel('model', size=14)
    plt.ylabel(metric, size=14)
    plt.title(f'{metric} distributions ({dataset} test)')
    plt.savefig(os.path.join(save_dir, f'{dataset.lower()}-violin-{metric.lower()}.png'))


def barplot_intersection(dataset, metric, save_dir=os.getcwd()):
    plt.figure()
    plt.rc('font', **font)
    data = {
        'model_type': [],
        metric: []
    }
    for file in os.listdir(result_dir):
        if not file.startswith(dataset.lower()):
            continue
        with open(os.path.join(result_dir, file), 'r') as fp:
            item = json.load(fp)
            data['model_type'].extend([item['model_type'] for _ in range(len(item[metric]))])
            data[metric].extend(item[metric])
    sns.barplot(x='model_type', y=metric, data=data)
    plt.xticks()
    plt.yticks()
    plt.xlabel('model', size=14)
    plt.ylabel(metric, size=14)
    plt.title(f'{metric} distributions ({dataset} test)')
    plt.savefig(os.path.join(save_dir, f'{dataset.lower()}-bar-{metric.lower()}.png'))


def scatter_dockq_compare(save_dir=os.getcwd()):
    plt.figure()
    plt.rc('font', **font)
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="grey")
    data = []
    for file in ['sabdab_ellidock.json', 'sabdab_multimer.json']:
        with open(os.path.join(result_dir, file), 'r') as fp:
            item = json.load(fp)
            data.append(item['DockQ'])
    sns.scatterplot(x=data[0], y=data[1], size=10, legend=False)
    plt.xticks()
    plt.yticks()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('ElliDock DockQ', size=14)
    plt.ylabel('Multimer DockQ', size=14)
    plt.title(f'ElliDock vs Multimer on DockQ (SAbDab test)')
    plt.savefig(os.path.join(save_dir, f'sabdab-dockq-compare.png'))


if __name__ == "__main__":
    fig_dir = './fig'
    os.makedirs(fig_dir, exist_ok=True)
    # scatterplot_irmsd_crmsd(dataset='db5')
    scatter_dockq_compare(save_dir=fig_dir)
    datasets = ['DB5', 'SAbDab']
    metrics = ['CRMSD', 'IRMSD', 'DockQ', 'intersection']
    for dataset in datasets:
        for metric in metrics:
            barplot_intersection(dataset, metric=metric, save_dir=fig_dir)
            violinplot(dataset=dataset, metric=metric, save_dir=fig_dir)
