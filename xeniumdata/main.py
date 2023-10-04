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
from .images import resize_image, register_image, fit_image_to_size_limit, deconvolve_he, write_ome_tiff
import cv2
import gc
import functools as ft
import seaborn as sns

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
    
    def __init__(self, 
                 path: Union[str, os.PathLike, Path],
                 metadata_filename: str = "experiment_modified.xenium",
                 transcript_filename: str = "transcripts.parquet",
                 pattern_xenium_folder: str = "output-{ins_id}__{slide_id}__{region_id}__{date}__{id}",
                 ):
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
        p_parsed = parse(pattern_xenium_folder, self.path.stem)
        self.slide_id = p_parsed.named["slide_id"]
        self.region_id = p_parsed.named["region_id"]
        
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
                        dapi_type: str = "focus",
                        pattern_img_file: str = "{slide_id}__{region_id}__{image_name}",
                        decon_scale_factor: float = 0.2
                        ):
        '''
        Register images stored in XeniumData object.
        '''
        # add arguments to object
        self.img_dir = Path(img_dir)
        self.pattern_img_file = pattern_img_file
        
        print("Processing path {}{}{}".format(tf.Bold, self.path, tf.ResetAll), flush=True)

        # get a list of image files
        img_files = sorted(self.img_dir.glob("*{}".format(img_suffix)))
        
        # find the corresponding image
        corr_img_files = [elem for elem in img_files if self.slide_id in str(elem) and self.region_id in str(elem)]
        
        # make sure images corresponding to the Xenium data were found
        if len(corr_img_files) == 0:
            print('\tNo image corresponding to the slide_id `{}` and the region_id `{}` were found.'.format(self.slide_id, self.region_id))
        else:
            if self.metadata_filename == "experiment_modified.xenium":
                print("\tFound modified `{}` file. Information will be added to this file.".format(self.metadata_filename))
            elif self.metadata_filename == "experiment.xenium":
                print("\tOnly unmodified metadata file (`{}`) found. Information will be added to new file (`{}`).".format(self.metadata_filename, "experiment_modified.xenium"))
            else:
                raise AssertionError("\tNo metadata file was found.")

            for img_file in corr_img_files:
                # parse name of current image
                img_stem = img_file.stem.split(".")[0] # make sure to remove also suffices like .ome.tif
                img_file_parsed = parse(pattern_img_file, img_stem)
                img_name = img_file_parsed.named["image_name"]
                print('\tProcessing {}"{}"{} image'.format(tf.Bold, img_name, tf.ResetAll), flush=True)
                
                # load registered image (usually DAPI in case of Xenium)
                relpath_template = self.metadata['images']['morphology_{}_filepath'.format(dapi_type)]
                path_template = Path(os.path.normpath(os.path.join(self.path, relpath_template))) # get the absolute path to the image

                print("\t\tLoading images...", flush=True)
                
                # read images
                image = imread(img_file)[0] # e.g. HE image
                self.read_images() # read DAPI image with XeniumData
                template = self.images.DAPI # usually DAPI image
                
                # load into memory
                image = image.compute()
                if isinstance(template, dask.array.core.Array):
                    template = template.compute()

                # resize image if necessary (warpAffine has a size limit for the image that is transformed)
                xy_shape_image = image.shape[:2]
                if np.any([elem > SHRT_MAX for elem in xy_shape_image]):
                    print(
                        "\t\tWarning: Dimensions of image ({}) exceed C++ limit SHRT_MAX ({}). " \
                        "Image dimensions are resized to meet requirements. This leads to a loss of quality.".format(image.shape, SHRT_MAX))
                    
                    # fit image
                    image_resized, sf_image = fit_image_to_size_limit(image, size_limit=SHRT_MAX, return_scale_factor=True)
                else:
                    sf_image = 1
                    image_resized = image

                print("\t\tRun color deconvolution", flush=True)
                # deconvolve HE - performed on resized image to save memory
                hema, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor), return_type="grayscale", convert=True)

                # bring back to original size
                hema_upsized = resize_image(hema, scale_factor=1/decon_scale_factor)

                print("\t\tExtract common features from image and template", flush=True)
                # perform registration to extract the common features ptsA and ptsB
                _, H, matchedVis, ptsA, ptsB = register_image(image=hema_upsized, 
                                                            template=template, 
                                                            maxpx=4000, 
                                                            perspective_transform=False, 
                                                            verbose=False, 
                                                            do_registration=False,
                                                            return_features=True
                                                            )
                
                # estimate affine transformation matrix
                print("\t\tEstimate affine transformation matrix for original image", flush=True)
                (H, mask) = cv2.estimateAffine2D(ptsA, ptsB)
                
                if sf_image != 1:
                    print("\t\tEstimate affine transformation matrix for resized image", flush=True)
                    ptsA *= sf_image # scale images features in case it was originally larger than the warpAffine limits
                    (H_resized, mask) = cv2.estimateAffine2D(ptsA, ptsB)
                else:
                    H_resized = H
                
                # Use the transformation matrix to register the original HE image
                print("\t\tDo registration", flush=True)
                (h, w) = template.shape[:2]
                registered = cv2.warpAffine(image_resized, H_resized, (w, h))

                # save as OME-TIFF
                outfile = self.path / "registered_{}.ome.tif".format(img_name)
                print("\t\tSave OME-TIFF to {}".format(outfile), flush=True)
                write_ome_tiff(
                    file=outfile, 
                    image=registered, 
                    axes="YXS", 
                    overwrite=True
                    )

                # save registration QC files
                reg_dir = self.path.parent / "registration"
                reg_dir.mkdir(parents=True, exist_ok=True) # create folder for QC outputs
                print("\t\tSave QC files to {}".format(reg_dir), flush=True)
                
                # save transformation matrix
                H = np.vstack([H, [0,0,1]]) # add last line of affine transformation matrix
                H_csv = reg_dir / "{}__{}__{}__H.csv".format(self.slide_id, self.region_id, img_name)
                np.savetxt(H_csv, H, delimiter=",") # save as .csv file
                
                # remove last line break from csv since this gives error when importing to Xenium Explorer
                remove_last_line_from_csv(H_csv)

                # save image showing the number of key points found in both images during registration
                matchedVis_file = reg_dir / "{}__{}__{}__matchedvis.pdf".format(self.slide_id, self.region_id, img_name)
                plt.imshow(matchedVis)
                plt.savefig(matchedVis_file, dpi=400)
                plt.close()

                # save metadata
                self.metadata['images']['registered_{}_filepath'.format(img_name)] = os.path.relpath(outfile, self.path)
                metadata_json = json.dumps(self.metadata, indent=4)
                metadata_mod_path = self.path / "experiment_modified.xenium"
                print("\t\tSave metadata to {}".format(metadata_mod_path), flush=True)
                with open(metadata_mod_path, "w") as metafile:
                    metafile.write(metadata_json)

                # free RAM
                del image, template, registered, hema_upsized, hema, eo, dab
                gc.collect()
        