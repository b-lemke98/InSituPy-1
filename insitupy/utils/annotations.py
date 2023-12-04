import pandas as pd
import numpy as np
from shapely import Point
from .utils import convert_to_list
import numpy as np
import pandas as pd
from .utils import textformat as tf

def assign_annotations(self, 
            annotation_labels: str = "all",
            add_annotation_masks: bool = False
            ):
    '''
    Function to assign the annotations to the anndata object in XeniumData.matrix.
    Annotation information is added to the DataFrame in `.obs`.
    '''
    # assert that prerequisites are met
    assert hasattr(self, "matrix"), "No .matrix attribute available. Run `read_matrix()`."
    assert hasattr(self, "annotations"), "No .matrix attribute available. Run `read_matrix()`."
    
    if annotation_labels == "all":
        annotation_labels = self.annotations.metadata.keys()
        
    # make sure annotation labels are a list
    annotation_labels = convert_to_list(annotation_labels)
    
    # convert coordinates into shapely Point objects
    points = [Point(elem) for elem in self.matrix.obsm["spatial"]]

    # iterate through annotation labels
    for annotation_label in annotation_labels:
        print(f"Assigning label '{annotation_label}'...")
        # extract pandas dataframe of current label
        annot = getattr(self.annotations, annotation_label)
        
        # get unique list of annotation names
        annot_names = annot.name.unique()
        
        # initiate dataframe as dictionary
        df = {}

        # iterate through names
        for n in annot_names:
            polygons = annot[annot.name == n].geometry.tolist()
            
            in_poly = []
            for poly in polygons:
                # check if which of the points are inside the current annotation polygon
                in_poly.append(poly.contains(points))
            
            # check if points were in any of the polygons
            in_poly_res = np.array(in_poly).any(axis=0)
            
            # collect results
            df[n] = in_poly_res
            
        # convert into pandas dataframe
        df = pd.DataFrame(df)
        df.index = self.matrix.obs_names
        
        # create annotation from annotation masks
        df[f"annotation-{annotation_label}"] = [" & ".join(annot_names[row.values]) if np.any(row.values) else np.nan for i, row in df.iterrows()]
        
        if add_annotation_masks:
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=df, left_index=True, right_index=True)
        else:
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=df.iloc[:, -1], left_index=True, right_index=True)
            
        # save that the current label was analyzed
        self.annotations.metadata[annotation_label]["analyzed"] = tf.TICK