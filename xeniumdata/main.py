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
from .images import resize_image, deconvolve_he, ImageRegistration
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
    from .utils.visualize import show
    
    # import crop function
    from .utils.crop import crop    
    
    def __init__(self, 
                 path: Union[str, os.PathLike, Path],
                 metadata_filename: str = "experiment_modified.xenium",
                 transcript_filename: str = "transcripts.parquet",
                 pattern_xenium_folder: str = "output-{ins_id}__{slide_id}__{region_id}",
                 matrix: Optional[AnnData] = None
                 ):
        if matrix is None:
            self.path = Path(path)
            self.transcript_filename = transcript_filename
            
            # check if path exists
            if not self.path.is_dir():
                raise FileNotFoundError(f"No such directory found: {str(self.path)}")
            
            # check for modified metadata_filename
            metadata_files = [elem.name for elem in self.path.glob("*.xenium")]
            if "experiment_modified.xenium" in metadata_files:
                self.metadata_filename = "experiment_modified.xenium"
            else:
                self.metadata_filename = "experiment.xenium"
                
            # all changes are saved to the modified .xenium json
            self.metadata_save_path = self.path / "experiment_modified.xenium"
                
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
    
    def copy(self):
        '''
        Function to generate a deep copy of the XeniumData object.
        '''
        from copy import deepcopy
        
        return deepcopy(self)
    
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
        
        
    def save_metadata(self,
                      metadata_path: Union[str, os.PathLike, Path] = None
                      ):
        # if there is no specific path given, the metadata is written to the default path for modified metadata
        if metadata_path is None:
            metadata_path = self.metadata_save_path
            
        # write to json file
        metadata_json = json.dumps(self.metadata, indent=4)
        print(f"\t\tSave metadata to {metadata_path}", flush=True)
        with open(metadata_path, "w") as metafile:
            metafile.write(metadata_json)


    def register_images(self,
                        img_dir: Union[str, os.PathLike, Path],
                        img_suffix: str = ".ome.tif",
                        pattern_img_file: str = "{slide_id}__{region_id}__{image_names}__{image_type}",
                        decon_scale_factor: float = 0.2,
                        image_name_sep: str = "_",  # string separating the image names in the file name
                        nuclei_name: str = "DAPI",  # name used for the nuclei image
                        #dapi_channel: int = None
                        ):
        '''
        Register images stored in XeniumData object.
        '''

        # add arguments to object
        self.img_dir = Path(img_dir)
        self.pattern_img_file = pattern_img_file
        
        # check if image path exists
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"No such directory found: {str(self.img_dir)}")
        
        print(f"Processing region {tf.Bold}{self.region_id}{tf.ResetAll} of slide {tf.Bold}{self.slide_id}{tf.ResetAll}", flush=True)        
        
        # get a list of image files
        img_files = sorted(self.img_dir.glob("*{}".format(img_suffix)))
        
        # find the corresponding image
        corr_img_files = [elem for elem in img_files if self.slide_id in str(elem) and self.region_id in str(elem)]
        
        # make sure images corresponding to the Xenium data were found
        if len(corr_img_files) == 0:
            print(f'\tNo image corresponding to slide`{self.slide_id}` and region `{self.region_id}` were found.')
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
                self.image_names = img_file_parsed.named["image_names"].split(image_name_sep)
                image_type = img_file_parsed.named["image_type"] # check which image type it has (`histo` or `IF`)
                
                # determine the structure of the image axes and check other things
                axes_template = "YX"
                if image_type == "histo":
                    axes_image = "YXS"
                    
                    # make sure that there is only one image name given
                    if len(self.image_names) > 1:
                        raise ValueError(f"More than one image name retrieved ({self.image_names})")
                    elif len(self.image_names) == 0:
                        raise ValueError(f"No image name found in file {img_file}")
                    else:
                        # in case of histo images only one image name should be given
                        self.image_names = self.image_names[0]
                    
                elif image_type == "IF":
                    axes_image = "CYX"
                else:
                    raise UnknownOptionError(image_type, available=["histo", "IF"])
                
                print(f'\tProcessing {tf.Bold}{self.image_names}{tf.ResetAll} {image_type} image', flush=True)

                # read images
                print("\t\tLoading images...", flush=True)
                image = imread(img_file) # e.g. HE image
                
                # sometimes images are read with an empty time dimension in the first axis. 
                # If this is the case, it is removed here.
                if len(image.shape) == 4:
                    image = image[0]
                    
                # read images in XeniumData object
                self.read_images()
                template = self.images.DAPI[0] # usually DAPI image. Use highest resolution of pyramid.
                
                # Setup image registration object - is important to load and scale the images.
                # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.
                imreg_complete = ImageRegistration(
                    image=image,
                    template=template,
                    axes_image=axes_image,
                    axes_template=axes_template,
                    verbose=False
                    )
                imreg_complete.load_and_scale_images()
                
                # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
                axes_selected = "YX" 
                if image_type == "histo":
                    print("\t\tRun color deconvolution", flush=True)
                    # deconvolve HE - performed on resized image to save memory
                    selected, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes=axes_selected), 
                                                return_type="grayscale", convert=True)

                    # bring back to original size
                    selected = resize_image(selected, scale_factor=1/decon_scale_factor, axes=axes_selected)
                    
                    # set nuclei_channel and nuclei_axis to None
                    nuclei_channel = channel_axis = None
                else:
                    # image_type is "IF" then
                    # get index of nuclei channel
                    nuclei_channel = self.image_names.index(nuclei_name)
                    channel_axis = axes_image.find("C")
                    
                    if channel_axis == -1:
                        raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")
                    
                    print(f"\t\tSelect image with nuclei from IF image (channel: {nuclei_channel})", flush=True)
                    # select nuclei channel from IF image
                    if nuclei_channel is None:
                        raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")
                    
                    # select dapi channel for registration
                    selected = np.take(image, nuclei_channel, channel_axis)
                    #selected = image[nuclei_channel]
                
                

                print("\t\tExtract common features from image and template", flush=True)
                # perform registration to extract the common features ptsA and ptsB
                imreg_selected = ImageRegistration(
                    image=selected,
                    template=imreg_complete.template,
                    axes_image=axes_selected,
                    axes_template=axes_template,
                    max_width=4000,
                    convert_to_grayscale=False,
                    perspective_transform=False
                )
                
                # run all steps to extract features and get transformation matrix
                imreg_selected.load_and_scale_images()
                imreg_selected.extract_features()
                imreg_selected.calculate_transformation_matrix()
                
                # Use the transformation matrix to register image to the template
                # First, determine if the image was resized before performing the steps above (necessary due to C++ limits in OpenCV)
                # Then, add the correct image to the ImageRegistration object that is used for registration
                # if hasattr(imreg_complete, "image_resized"):
                #     imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image
                #     # _image = imreg_complete.image_resized # use resized original image
                #     # _T = imreg_selected.T_resized # use resized transformation matrix generated with selected image
                # else:
                #     assert not hasattr(imreg_selected, "image_resized")
                #     imreg_selected.image = imreg_complete.image  # use original image
                    # _image = imreg_complete.image # use original image
                    # _T = imreg_selected.T # use original transformation matrix generated with selected image
                
                # print("\t\tDo registration", flush=True)
                # if imreg_selected.flip_axis is not None:
                #     print(f"\t\tImage is flipped {'vertically' if imreg_selected.flip_axis == 0 else 'horizontally'}", flush=True)
                #     _image = np.flip(_image, axis=imreg_selected.flip_axis)
                # (h, w) = template.shape[:2]
                
                # setup path for metadata
                metadata_mod_path = self.path / "experiment_modified.xenium"
                if image_type == "histo":
                    # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
                    #registered = cv2.warpAffine(_image, _T, (w, h))
                    
                    if hasattr(imreg_complete, "image_resized"):
                        imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image
                    else:
                        assert not hasattr(imreg_selected, "image_resized")
                        imreg_selected.image = imreg_complete.image  # use original image
                    
                    # perform registration
                    imreg_selected.perform_registration()
                        
                    # save files
                    imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.region_id}__{self.image_names}",
                                        axes=axes_image,
                                        photometric='rgb'
                                        )
                    
                    # save metadata
                    self.metadata['images'][f'registered_{self.image_names}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path)
                    self.save_metadata()
                        
                    del imreg_complete, imreg_selected, image, template, selected, eo, dab
                else:
                    # image_type is IF
                    # In case of IF images the channels are normally in the first axis and each channel is registered separately
                    # Further, each channel is then saved separately as grayscale image.
                    
                    # iterate over channels
                    for i, n in enumerate(self.image_names):
                        # skip the DAPI image
                        if n == nuclei_name:
                            break
                        
                        if hasattr(imreg_complete, "image_resized"):
                            # select one channel from resized original image
                            imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)
                        else:
                            assert not hasattr(imreg_selected, "image_resized")
                            # select one channel from non-resized original image
                            imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
                            
                        # perform registration
                        imreg_selected.perform_registration()
                        
                        # save files
                        imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.region_id}__{n}",
                                        axes='YX',
                                        photometric='minisblack'
                                        )
                        
                        # save metadata
                        self.metadata['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path)
                        self.save_metadata()

                    # free RAM
                    del imreg_complete, imreg_selected, image, template, selected
                gc.collect()
