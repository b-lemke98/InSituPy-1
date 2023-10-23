import json
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask_image.imread import imread
import dask
from .utils.utils import textformat as tf
from .utils.utils import remove_last_line_from_csv
from parse import *
from .images import resize_image, deconvolve_he, write_ome_tiff, ImageRegistration
import cv2
import gc
import functools as ft
import seaborn as sns
from anndata import AnnData
from .utils.exceptions import UnknownOptionError

# make sure that image does not exceed limits in c++ (required for cv2::remap function in cv2::warpAffine)
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # -32767

def read_xenium_metadata(
    path: Union[str, os.PathLike, Path],
    metadata_filename: str = "experiment.xenium"
    ) -> dict:
    '''
    Function to read the xenium metadata file which usually is in the xenium output folder of one region.
    '''
    # load metadata file
    metapath = path / metadata_filename
    with open(metapath, "r") as metafile:
        metadata = json.load(metafile)
        
    return metadata

class XeniumData:
    '''
    XeniumData object to read Xenium in situ data in a structured way.
    '''
    # import read functions
    from .utils.read import read_all, read_annotations, read_boundaries, read_images, read_matrix, read_transcripts
    
    # import analysis functions
    from .utils.annotations import annotate
    
    # import preprocessing functions
    from .utils.preprocessing import normalize, hvg, reduce_dimensions
    
    # import visualization functions
    from .utils.visualize import interactive
    
    # import crop function
    from .utils.crop import crop
    
    def __init__(self, 
                 path: Optional[Union[str, os.PathLike, Path]],
                 metadata_filename: str = "experiment_modified.xenium",
                 transcript_filename: str = "transcripts.parquet",
                 pattern_xenium_folder: str = "output-{ins_id}__{slide_id}__{region_id}",
                 matrix: Optional[AnnData] = None
                 ):
        if matrix is None:
            self.path = Path(path)
            self.transcript_filename = transcript_filename
            
            # check for modified metadata_filename
            metadata_files = [elem.name for elem in self.path.glob("*.xenium")]
            if "experiment_modified.xenium" in metadata_files:
                self.metadata_filename = "experiment_modified.xenium"
            else:
                self.metadata_filename = "experiment.xenium"
                
            # read metadata
            self.metadata = read_xenium_metadata(self.path, metadata_filename=self.metadata_filename)
            
            # parse folder name to get slide_id and region_id
            name_stub = "__".join(self.path.stem.split("__")[:3])
            p_parsed = parse(pattern_xenium_folder, name_stub)
            self.slide_id = p_parsed.named["slide_id"]
            self.region_id = p_parsed.named["region_id"]
        else:
            self.matrix = matrix
            self.slide_id = ""
            self.region_id = ""
            self.path = Path("unknown/unknown")
            self.metadata_filename = ""
        
    def __repr__(self):
        repr = (
            f"{tf.Bold+tf.Red}XeniumData{tf.ResetAll}\n" 
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self.slide_id}\n"
            f"{tf.Bold}Region ID:{tf.ResetAll}\t{self.region_id}\n"
            f"{tf.Bold}Data path:{tf.ResetAll}\t{self.path.parent}\n"
            f"{tf.Bold}Data folder:{tf.ResetAll}\t{self.path.name}\n"
            f"{tf.Bold}Metadata file:{tf.ResetAll}\t{self.metadata_filename}"            
        )
        
        if hasattr(self, "images"):
            images_repr = self.images.__repr__()
            repr = (
                #repr + f"\n{tf.Bold}Images:{tf.ResetAll} "
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + images_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
            
        if hasattr(self, "matrix"):
            matrix_repr = self.matrix.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} matrix{tf.ResetAll}\n{tf.SPACER}   " + matrix_repr.replace("\n", "\n\t   ")
            )
        
        if hasattr(self, "transcripts"):
            trans_repr = f"DataFrame with shape {self.transcripts.shape[0]} x {self.transcripts.shape[1]}"
            
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.LightCyan+tf.Bold} transcripts{tf.ResetAll}\n\t   " + trans_repr
            )
            
        if hasattr(self, "boundaries"):
            bound_repr = self.boundaries.__repr__()
            repr = (
                #repr + f"\n{tf.Bold}Images:{tf.ResetAll} "
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + bound_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
            
        if hasattr(self, "annotations"):
            annot_repr = self.annotations.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + annot_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
        return repr
    
    def plot_dimred(self, save: Optional[str] = None):
        '''
        Read dimensionality reduction plots.
        '''
        # construct paths
        analysis_path = self.path / "analysis"
        umap_file = analysis_path / "umap" / "gene_expression_2_components" / "projection.csv"
        pca_file = analysis_path / "pca" / "gene_expression_10_components" / "projection.csv"
        cluster_file = analysis_path / "clustering" / "gene_expression_graphclust" / "clusters.csv"
        
        
        # read data
        umap_data = pd.read_csv(umap_file)
        pca_data = pd.read_csv(pca_file)
        cluster_data = pd.read_csv(cluster_file)
        
        # merge dimred data with clustering data
        data = ft.Reduce(lambda left, right: pd.merge(left, right, on='Barcode'), [umap_data, pca_data.iloc[:, :3], cluster_data])
        data["Cluster"] = data["Cluster"].astype('category')
                        
        # plot
        nrows = 1
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        sns.scatterplot(data=data, x="PC-1", y="PC-2", hue="Cluster", palette="tab20", ax=axs[0])
        sns.scatterplot(data=data, x="UMAP-1", y="UMAP-2", hue="Cluster", palette="tab20", ax=axs[1])
        if save is not None:
            plt.savefig(save)
        plt.show()


    def register_images(self,
                        img_dir: Union[str, os.PathLike, Path],
                        img_suffix: str = ".ome.tif",
                        pattern_img_file: str = "{slide_id}__{region_id}__{image_name}__{image_type}",
                        decon_scale_factor: float = 0.2,
                        dapi_channel: int = None
                        ):
        '''
        Register images stored in XeniumData object.
        '''

        # add arguments to object
        self.img_dir = Path(img_dir)
        self.pattern_img_file = pattern_img_file
        
        print(f"Processing path {tf.Bold}{self.path}{tf.ResetAll}", flush=True)

        # get a list of image files
        img_files = sorted(self.img_dir.glob("*{}".format(img_suffix)))
        
        # find the corresponding image
        corr_img_files = [elem for elem in img_files if self.slide_id in str(elem) and self.region_id in str(elem)]
        
        # make sure images corresponding to the Xenium data were found
        if len(corr_img_files) == 0:
            print(f'\tNo image corresponding to the slide_id `{self.slide_id}` and the region_id `{self.region_id}` were found.')
        else:
            if self.metadata_filename == "experiment_modified.xenium":
                print(f"\tFound modified `{self.metadata_filename}` file. Information will be added to this file.")
            elif self.metadata_filename == "experiment.xenium":
                print(f"\tOnly unmodified metadata file (`{self.metadata_filename}`) found. Information will be added to new file (`experiment_modified.xenium`).")
            else:
                raise FileNotFoundError("Metadata file not found.")

            for img_file in corr_img_files:
                # parse name of current image
                img_stem = img_file.stem.split(".")[0] # make sure to remove also suffices like .ome.tif
                img_file_parsed = parse(pattern_img_file, img_stem)
                image_name = img_file_parsed.named["image_name"]
                image_type = img_file_parsed.named["image_type"] # check which image type it has (`histo` or `IF`)
                
                img_type_options = ["histo", "IF"]
                if image_type not in img_type_options:
                    raise UnknownOptionError(image_type, available=img_type_options)
                
                print(f'\tProcessing {tf.Bold}"{image_name}"{tf.ResetAll} {image_type} image', flush=True)
                # load registered image (usually DAPI in case of Xenium)
                #relpath_template = self.metadata['images'][f'morphology_{dapi_type}_filepath']
                #path_template = Path(os.path.normpath(os.path.join(self.path, relpath_template))) # get the absolute path to the image

                # read images
                print("\t\tLoading images...", flush=True)
                image = imread(img_file) # e.g. HE image
                
                # sometimes images are read with a leading time dimension. If this is the case, it is removed here.
                if len(image.shape) == 4:
                    image = image[0]
                    
                # read images in XeniumData object
                self.read_images()
                template = self.images.DAPI[0] # usually DAPI image. Use highest resolution from pyramid.
                
                # setup image registration object - is important to load and scale the images
                imreg_complete = ImageRegistration(
                    image=image,
                    template=template,
                    verbose=False
                    )
                imreg_complete.load_and_scale_images()

                if image_type == "histo":
                    print("\t\tRun color deconvolution", flush=True)
                    # deconvolve HE - performed on resized image to save memory
                    selected, eo, dab = deconvolve_he(img=resize_image(image, 
                                                                scale_factor=decon_scale_factor), 
                                                return_type="grayscale", convert=True)

                    # bring back to original size
                    selected = resize_image(selected, scale_factor=1/decon_scale_factor)
                else:
                    print(f"\t\tSelect DAPI image from IF image (channel: {dapi_channel})", flush=True)
                    # select DAPI channel from IF image
                    if dapi_channel is None:
                        raise TypeError("Argument `dapi_channel` should be an integer and not NoneType.")
                    
                    # select dapi channel for registration
                    selected = image[dapi_channel]

                print("\t\tExtract common features from image and template", flush=True)
                # perform registration to extract the common features ptsA and ptsB
                imreg_selected = ImageRegistration(
                    image=selected,
                    template=imreg_complete.template,
                    max_width=4000,
                    convert_to_grayscale=False,
                    perspective_transform=False
                )
                
                # run all steps to extract features and get transformation matrix
                imreg_selected.load_and_scale_images()
                imreg_selected.extract_features()
                imreg_selected.calculate_transformation_matrix()
                
                # Use the transformation matrix to register the original HE image
                # determine which image to be registered
                if hasattr(imreg_complete, "image_resized"):
                    _image = imreg_complete.image_resized # use resized original image
                    _T = imreg_selected.T_resized # use resized transformation matrix generated with selected image
                else:
                    _image = imreg_complete.image # use original image
                    _T = imreg_selected.T # use original transformation matrix generated with selected image
                
                print("\t\tDo registration", flush=True)
                if imreg_selected.flip_axis is not None:
                    print(f"\t\tImage is flipped {'vertically' if imreg_selected.flip_axis == 0 else 'horizontally'}", flush=True)
                    _image = np.flip(_image, axis=imreg_selected.flip_axis)
                (h, w) = template.shape[:2]
                self.registered = cv2.warpAffine(_image, _T, (w, h))

                # save as OME-TIFF
                outfile = self.path / f"registered_{image_name}.ome.tif"
                print(f"\t\tSave OME-TIFF to {outfile}", flush=True)
                write_ome_tiff(
                    file=outfile, 
                    image=self.registered, 
                    axes="YXS", 
                    overwrite=True
                    )

                # save registration QC files
                reg_dir = self.path.parent / "registration"
                reg_dir.mkdir(parents=True, exist_ok=True) # create folder for QC outputs
                print(f"\t\tSave QC files to {reg_dir}", flush=True)
                
                # save transformation matrix
                T = np.vstack([_T, [0,0,1]]) # add last line of affine transformation matrix
                T_csv = reg_dir / f"{self.slide_id}__{self.region_id}__{image_name}__H.csv"
                np.savetxt(T_csv, T, delimiter=",") # save as .csv file
                
                # remove last line break from csv since this gives error when importing to Xenium Explorer
                remove_last_line_from_csv(T_csv)

                # save image showing the number of key points found in both images during registration
                matchedVis_file = reg_dir / f"{self.slide_id}__{self.region_id}__{image_name}__matchedvis.pdf"
                plt.imshow(imreg_selected.matchedVis)
                plt.savefig(matchedVis_file, dpi=400)
                plt.close()

                # save metadata
                self.metadata['images'][f'registered_{image_name}_filepath'] = os.path.relpath(outfile, self.path)
                metadata_json = json.dumps(self.metadata, indent=4)
                metadata_mod_path = self.path / "experiment_modified.xenium"
                print(f"\t\tSave metadata to {metadata_mod_path}", flush=True)
                with open(metadata_mod_path, "w") as metafile:
                    metafile.write(metadata_json)

                # free RAM
                del imreg_complete, imreg_selected, image, template, selected, eo, dab
                gc.collect()
        