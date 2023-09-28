from pathlib import Path
import pandas as pd
import os

class XeniumPanels:
    '''
    Class containing a collection of custom color palettes.
    '''
    def __init__(self, verbose=False):
        # read all panels
        script_dir = Path(os.path.realpath(__file__)).parent
        panel_dir = script_dir / Path("../xenium_panels/")
        panel_dir = panel_dir.resolve()
        panel_paths = sorted(panel_dir.glob("*.csv"))

        for p in panel_paths:
            name = p.stem.split("_")[1]
            panel = pd.read_csv(p)
            setattr(self, name, panel)
            print(name) if verbose else None

            # make sure that the column names are correct
            panel.columns = ["Gene", "Ensembl_ID", "Coverage", "Codewords", "Annotation"]
        
    def show_all(self):
        '''
        Prints all available panels.
        '''
        panel_list = [elem for elem in dir(self) if not elem.startswith("__")]
        panel_list = [elem for elem in panel_list if elem not in ["show_all"]]
        for p in panel_list:
            print(p)