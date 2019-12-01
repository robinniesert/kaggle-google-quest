import matplotlib.pyplot as plt


def twin_plot(vals1, vals2, ax1_label=None, ax2_label=None, 
              y_label=None, label1=None, label2=None):
    _, ax1 = plt.subplots()
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(ax1_label)
    ax1.plot(vals1, label=label1)

    ax2 = ax1.twiny()
    ax2.set_xlabel(ax2_label)
    ax2.plot(vals2, c='C1', label=label2)

    ln1, lab1 = ax1.get_legend_handles_labels()
    ln2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(ln1+ln2, lab1+lab2)