import matplotlib.pyplot as plt

from common import N_CLASSES, CLASSES

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

          
def visualize_with_raw(image, mask, original_image=None, original_mask=None, 
                       raw_image=None, raw_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    _, ax = plt.subplots(3, N_CLASSES+1, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i, class_id in enumerate(CLASSES):
        ax[0, i+1].imshow(original_mask[:, :, i])
        ax[0, i+1].set_title(f'Original mask {class_id}', fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i, class_id in enumerate(CLASSES):
        ax[1, i+1].imshow(raw_mask[:, :, i])
        ax[1, i+1].set_title(f'Raw predicted mask {class_id}', fontsize=fontsize)
        
    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)

    for i, class_id in enumerate(CLASSES):
        ax[2, i+1].imshow(mask[:, :, i])
        ax[2, i+1].set_title(
            f'Predicted mask with processing {class_id}', fontsize=fontsize)