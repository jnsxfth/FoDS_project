from script import *


#Function to plot the variable as histogram
def plot_data(data, column):
    plt.figure(figsize=(10, 8))
    sns.histplot(data[column], binwidth=1, discrete=True, color='#008F91')
    plt.title(f"Distribution of\n{titles[column]} in {get_dataset_name(data)}", size=12, fontweight='bold')
    plt.xlabel(xlabels[column], fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.savefig(f'./output/{get_dataset_name(data)}/{column}_histplot')
    plt.close()


#Function to plot the variable as boxplot
def boxplot_data(data, column):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=data[column], color='#008F91')
    plt.title(f'Distribution of\n {titles[column]} in {get_dataset_name(data)}', size=12, fontweight='bold')
    plt.xlabel(xlabels[column], fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.savefig(f'./output/{get_dataset_name(data)}/{column}_boxplot')
    plt.close()
