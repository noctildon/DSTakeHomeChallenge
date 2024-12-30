import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def histplot_ratio(sdata, figsize=(25, 6), title='ratio', color='lightblue'):
    """

    eg.
    sdata = sns.histplot(data=data, x='timestamp', hue='converted', bins=30);
    histplot_ratio(sdata)
    see example in 6_Pricing_Test.ipynb
    """

    legend_labels = [text.get_text() for text in sdata.legend_.get_texts()]
    hist_data = {label: [] for label in legend_labels}
    for patch in sdata.patches:
        color = patch.get_facecolor()
        for label, legend_patch in zip(legend_labels, sdata.legend_.get_patches()):
            if np.allclose(color, legend_patch.get_facecolor()):
                bin_start = patch.get_x()
                bin_end = bin_start + patch.get_width()
                height = patch.get_height()

                bin_start = pd.Timestamp(bin_start, unit='D')
                bin_end = pd.Timestamp(bin_end, unit='D')
                hist_data[label].append({"bin_start": bin_start, "bin_end": bin_end, "frequency": height})

    hist_bins_data = {'edges':[] , 'ratio': []}

    # put bin_start and bin_end into a list
    for bin_data0, bin_data1 in zip(hist_data['0'], hist_data['1']):
        hist_bins_data['edges'].append(bin_data0['bin_start'])
        hist_bins_data['ratio'].append(bin_data1['frequency']/bin_data0['frequency'])

    hist_bins_data['edges'].append(bin_data0['bin_end'])


    plt.figure(figsize=figsize)
    for i in range(len(hist_bins_data['edges'])-1):
        width = (hist_bins_data['edges'][i+1] - hist_bins_data['edges'][i]).days
        plt.bar(hist_bins_data['edges'][i], hist_bins_data['ratio'][i], width=width, alpha=0.5, color=color)
    plt.title(title)
    plt.show()
