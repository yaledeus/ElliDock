import os
import sys
import json

import seaborn as sns
import matplotlib.pyplot as plt


def scatterplot_irmsd_crmsd(files, save_dir=os.getcwd()):
    plt.figure()
    for file in files:
        with open(file, 'r') as fp:
            data = json.load(fp)
            size = [50 - irmsd * crmsd for irmsd, crmsd in zip(data['IRMSD'], data['CRMSD'])]
            sns.scatterplot(x='IRMSD', y='CRMSD', size=size, label=data['model_type'], data=data, legend=False)
    plt.xlabel('IRMSD')
    plt.ylabel('CRMSD')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'scatterplot.png'))


if __name__ == "__main__":
    files = sys.argv[1:]
    scatterplot_irmsd_crmsd(files)
