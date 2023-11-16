import json
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask_image.imread import imread
from .utils.utils import textformat as tf
from .utils.utils import read_json
from parse import *
from .images import resize_image, deconvolve_he, ImageRegistration
import gc
import functools as ft
import seaborn as sns
from anndata import AnnData
from .utils.exceptions import UnknownOptionError

# make sure that image does not exceed limits in c++ (required for cv2::remap function in cv2::warpAffine)
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # -32767

class XeniumData:
    #TODO: Docstring of XeniumData
    """_summary_

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        ValueError: _description_
        UnknownOptionError: _description_
        ValueError: _description_
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    # import read and write functions
    from .io._read import read_all, read_annotations, read_boundaries, read_images, read_matrix, read_transcripts
    
    # import write function
    from .io._write import save
    
    # import analysis functions
    from .utils.annotations import annotate
    
    # import preprocessing functions
    from .utils.preprocessing import normalize, hvg, reduce_dimensions
    
    # import visualization functions
    from .visualize.visualize import show
    
    # import crop function
    from .utils.crop import crop
    
    def __init__(self, 
                 path: Union[str, os.PathLike, Path],
                 pattern_xenium_folder: str = "output-{ins_id}__{slide_id}__{sample_id}",
                 matrix: Optional[AnnData] = None
                 ):
        """_summary_

        Args:
            path (Union[str, os.PathLike, Path]): _description_
            pattern_xenium_folder (str, optional): _description_. Defaults to "output-{ins_id}__{slide_id}__{sample_id}".
            matrix (Optional[AnnData], optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
        """
        path = Path(path) # make sure the path is a pathlib path
        self.from_xeniumdata = False  # flag indicating from where the data is read
        if (path / "xeniumdata.json").is_file():
            self.path = Path(path)
            
            # read xeniumdata metadata
            self.metadata_filename = "xeniumdata.json"
            self.xd_metadata = read_json(self.path / self.metadata_filename)
            
            # read general xenium metadata
            self.metadata = read_json(self.path / "xenium.json")
            
            # retrieve slide_id and sample_id
            self.slide_id = self.xd_metadata["slide_id"]
            self.sample_id = self.xd_metadata["sample_id"]
            
            # set flag for xeniumdata
            self.from_xeniumdata = True
            
        elif matrix is None:
            self.path = Path(path)
            
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
            self.metadata = read_json(self.path / self.metadata_filename)
            
            # parse folder name to get slide_id and sample_id
            name_stub = "__".join(self.path.stem.split("__")[:3])
            p_parsed = parse(pattern_xenium_folder, name_stub)
            self.slide_id = p_parsed.named["slide_id"]
            self.sample_id = p_parsed.named["sample_id"]
        else:
            self.matrix = matrix
            self.slide_id = ""
            self.sample_id = ""
            self.path = Path("unknown/unknown")
            self.metadata_filename = ""
        
    def __repr__(self):
        repr = (
            f"{tf.Bold+tf.Red}XeniumData{tf.ResetAll}\n"
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self.slide_id}\n"
            f"{tf.Bold}Sample ID:{tf.ResetAll}\t{self.sample_id}\n"
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
        
        if hasattr(self, "viewer"):
            del self.viewer
        
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
        data = ft.reduce(lambda left, right: pd.merge(left, right, on='Barcode'), [umap_data, pca_data.iloc[:, :3], cluster_data])
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
                        pattern_img_file: str = "{slide_id}__{sample_id}__{image_names}__{image_type}",
                        decon_scale_factor: float = 0.2,
                        image_name_sep: str = "_",  # string separating the image names in the file name
                        nuclei_name: str = "DAPI",  # name used for the nuclei image
                        physicalsize: str = 'µm',
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
        
        print(f"Processing sample {tf.Bold}{self.sample_id}{tf.ResetAll} of slide {tf.Bold}{self.slide_id}{tf.ResetAll}", flush=True)        
        
        # get a list of image files
        img_files = sorted(self.img_dir.glob("*{}".format(img_suffix)))
        
        # find the corresponding image
        corr_img_files = [elem for elem in img_files if self.slide_id in str(elem) and self.sample_id in str(elem)]
        
        # make sure images corresponding to the Xenium data were found
        if len(corr_img_files) == 0:
            print(f'\tNo image corresponding to slide `{self.slide_id}` and sample `{self.sample_id}` were found.')
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
                    
                    if len(self.image_names) == 0:
                        raise ValueError(f"No image name found in file {img_file}")
                    
                elif image_type == "IF":
                    axes_image = "CYX"
                else:
                    raise UnknownOptionError(image_type, available=["histo", "IF"])
                
                print(f'\tProcessing following {image_type} images: {tf.Bold}{", ".join(self.image_names)}{tf.ResetAll}', flush=True)

                # read images
                print("\t\tLoading images to be registered...", flush=True)
                image = imread(img_file) # e.g. HE image
                
                # sometimes images are read with an empty time dimension in the first axis. 
                # If this is the case, it is removed here.
                if len(image.shape) == 4:
                    image = image[0]
                    
                # read images in XeniumData object
                self.read_images(names="nuclei")
                template = self.images.nuclei[0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.
                
                # extract OME metadata
                ome_metadata_template = self.images.metadata["nuclei"]["OME"]
                # extract pixel size for x and y from metadata
                pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}
                
                # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
                axes_selected = "YX" 
                if image_type == "histo":
                    print("\t\tRun color deconvolution", flush=True)
                    # deconvolve HE - performed on resized image to save memory
                    # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
                    nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes=axes_selected), 
                                                return_type="grayscale", convert=True)

                    # bring back to original size
                    nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes=axes_selected)
                    
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
                    nuclei_img = np.take(image, nuclei_channel, channel_axis)
                    #selected = image[nuclei_channel]
                    
                # Setup image registration objects - is important to load and scale the images.
                # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.
                
                # First: Setup the ImageRegistration object for the whole image (before deconvolution in histo images and multi-channel in IF)
                imreg_complete = ImageRegistration(
                    image=image,
                    template=template,
                    axes_image=axes_image,
                    axes_template=axes_template,
                    verbose=False
                    )
                # load and scale the whole image
                imreg_complete.load_and_scale_images()

                # setup ImageRegistration object with the nucleus image (either from deconvolution or just selected from IF image)
                imreg_selected = ImageRegistration(
                    image=nuclei_img,
                    template=imreg_complete.template,
                    axes_image=axes_selected,
                    axes_template=axes_template,
                    max_width=4000,
                    convert_to_grayscale=False,
                    perspective_transform=False
                )
                
                # run all steps to extract features and get transformation matrix
                imreg_selected.load_and_scale_images()
                
                print("\t\tExtract common features from image and template", flush=True)
                # perform registration to extract the common features ptsA and ptsB
                imreg_selected.extract_features()
                imreg_selected.calculate_transformation_matrix()
                
                if image_type == "histo":
                    # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
                    if imreg_complete.image_resized is None:
                        imreg_selected.image = imreg_complete.image  # use original image
                    else:
                        imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image
                    
                    # perform registration
                    imreg_selected.perform_registration()
                    
                    # generate OME metadata for saving
                    metadata = {
                        **{'SignificantBits': 8,
                           'PhysicalSizeXUnit': 'µm',
                           'PhysicalSizeYUnit': 'µm'
                         },
                        **pixelsizes
                    }
                        
                    # save files
                    imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.sample_id}__{self.image_names[0]}",
                                        axes=axes_image,
                                        photometric='rgb',
                                        ome_metadata=metadata
                                        )
                    
                    # save metadata
                    self.metadata['images'][f'registered_{self.image_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path)
                    self.save_metadata()
                        
                    del imreg_complete, imreg_selected, image, template, nuclei_img, eo, dab
                else:
                    # image_type is IF
                    # In case of IF images the channels are normally in the first axis and each channel is registered separately
                    # Further, each channel is then saved separately as grayscale image.
                    
                    # iterate over channels
                    for i, n in enumerate(self.image_names):
                        # skip the DAPI image
                        if n == nuclei_name:
                            break
                        
                        if imreg_complete.image_resized is None:
                            # select one channel from non-resized original image
                            imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
                        else:
                            # select one channel from resized original image
                            imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)
                            
                        # perform registration
                        imreg_selected.perform_registration()
                        
                        # save files
                        imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.sample_id}__{n}",
                                        axes='YX',
                                        photometric='minisblack'
                                        )
                        
                        # save metadata
                        self.metadata['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path)
                        self.save_metadata()

                    # free RAM
                    del imreg_complete, imreg_selected, image, template, nuclei_img
                gc.collect()
