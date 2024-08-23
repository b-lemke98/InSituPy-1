import os
from pathlib import Path

import matplotlib.pyplot as plt


def save_and_show_figure(savepath, fig, save_only=False, show=True, dpi_save=300, save_background=None, tight=True):
    #if fig is not None and axis is not None:
    #    return fig, axis
    #elif savepath is not None:
    if tight:
        fig.tight_layout()

    if savepath is not None:
        print("Saving figure to file " + savepath)

        # create path if it does not exist
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)

        # save figure
        plt.savefig(savepath, dpi=dpi_save,
                    facecolor=save_background, bbox_inches='tight')
        print("Saved.")
    if save_only:
        plt.close(fig)
    elif show:
        return plt.show()
    else:
        return