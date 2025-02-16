import gc
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from dask_image.imread import imread
from parse import *

from insitupy import __version__
from insitupy._core.insitudata import InSituData
from insitupy._exceptions import UnknownOptionError
from insitupy.images import (ImageRegistration, clip_image_histogram,
                             deconvolve_he, resize_image)
from insitupy.images.utils import otsu_thresholding
from insitupy.io.files import write_dict_to_json
from insitupy.utils.utils import convert_to_list
from insitupy.utils.utils import textformat as tf


def register_images(
    data: InSituData,
    image_to_be_registered: Union[str, os.PathLike, Path],
    image_type: Literal["histo", "IF"],
    channel_names: Union[str, List[str]],
    channel_name_for_registration: Optional[str] = None,  # name used for the nuclei image. Only required for IF images.
    template_image_name: str = "nuclei",
    save_registered_images: bool = True,
    min_good_matches_per_area: int = 5, # unit: 1/mm²
    test_flipping: bool = True,
    decon_scale_factor: float = 0.2,
    physicalsize: str = 'µm',
    prefix: str = "",
    ):
    """
    Register images stored in an InSituData object.

    Args:
        data (InSituData): The InSituData object containing the images.
        image_to_be_registered (Union[str, os.PathLike, Path]): Path to the image to be registered.
        image_type (Literal["histo", "IF"]): Type of the image, either "histo" or "IF".
        channel_names (Union[str, List[str]]): Names of the channels in the image.
        channel_name_for_registration (Optional[str], optional): Name of the channel used for registration. Required for IF images. Defaults to None.
        template_image_name (str, optional): Name of the template image. Defaults to "nuclei".
        save_registered_images (bool, optional): Whether to save the registered images. Defaults to True.
        min_good_matches (int, optional): Minimum number of good matches required for registration. Defaults to 20.
        test_flipping (bool): Whether to test flipping of images during registration. Defaults to True.
        decon_scale_factor (float, optional): Scale factor for deconvolution. Defaults to 0.2.
        physicalsize (str, optional): Unit of physical size. Defaults to 'µm'.
        prefix (str, optional): Prefix for the output files. Defaults to "".

    Raises:
        ValueError: If `image_type` is "IF" and `channel_name_for_registration` is None.
        FileNotFoundError: If the image to be registered is not found.
        ValueError: If more than one image name is retrieved for histo images.
        ValueError: If no image name is found in the file.
        UnknownOptionError: If an unknown image type is provided.
        TypeError: If `channel_name_for_registration` is None for IF images.
        ValueError: If no channel indicator `C` is found in the image axes.

    Returns:
        None
    """

    # if image type is IF, the channel name for registration needs to be given
    if image_type == "IF" and channel_name_for_registration is None:
        raise ValueError(f'If `image_type" is "IF", `channel_name_for_registration is not allowed to be `None`.')

    # define output directory
    output_dir = data.path.parent / "registered_images"

    # if output_dir.is_dir() and not force:
    #     raise FileExistsError(f"Output directory {output_dir} exists already. If you still want to run the registration, set `force=True`.")

    # check if image path exists
    image_to_be_registered = Path(image_to_be_registered)
    if not image_to_be_registered.is_file():
        raise FileNotFoundError(f"No such file found: {str(image_to_be_registered)}")

    # make sure the given image names are in a list
    channel_names = convert_to_list(channel_names)

    # determine the structure of the image axes and check other things
    axes_template = "YX"
    if image_type == "histo":
        axes_image = "YXS"

        # make sure that there is only one image name given
        if len(channel_names) > 1:
            raise ValueError(f"More than one image name retrieved ({channel_names})")

        if len(channel_names) == 0:
            raise ValueError(f"No image name found in file {image_to_be_registered}")

    elif image_type == "IF":
        axes_image = "CYX"
    else:
        raise UnknownOptionError(image_type, available=["histo", "IF"])

    print(f'\tProcessing following {image_type} images: {tf.Bold}{", ".join(channel_names)}{tf.ResetAll}', flush=True)

    # read images
    print("\t\tLoading images to be registered...", flush=True)
    image = imread(image_to_be_registered) # e.g. HE image

    # sometimes images are read with an empty time dimension in the first axis.
    # If this is the case, it is removed here.
    if len(image.shape) == 4:
        image = image[0]

    # # read images in InSituData object
    template = data.images[template_image_name][0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.

    # extract OME metadata
    #ome_metadata_template = data.images.metadata[template_image_name]["OME"]

    # get pixel size from image metadata
    pixel_size = data.images.metadata[template_image_name]["pixel_size"]

    # extract pixel size for x and y from OME metadata
    #pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}

    # generate OME metadata for saving
    ome_metadata = {
        'SignificantBits': 8,
        'PhysicalSizeXUnit': physicalsize,
        'PhysicalSizeYUnit': physicalsize,
        'PhysicalSizeX': pixel_size,
        'PhysicalSizeY': pixel_size
        }

    # determine minimum number of good matches that are necessary for the registration to be performed
    h, w = template.shape[:2]
    image_area = h * w * pixel_size**2 / 1000**2 # in mm²
    min_good_matches = int(min_good_matches_per_area * image_area)

    # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
    #axes_selected = "YX"
    if image_type == "histo":
        print("\t\tRun color deconvolution", flush=True)
        # deconvolve HE - performed on resized image to save memory
        # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
        nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes="YXS"),
                                    return_type="grayscale", convert=True)

        # bring back to original size
        nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes="YX")

        # set nuclei_channel and nuclei_axis to None
        channel_name_for_registration = channel_axis = None
    else:
        # image_type is "IF" then
        # get index of nuclei channel
        channel_id_for_registration = channel_names.index(channel_name_for_registration)
        channel_axis = axes_image.find("C")

        if channel_axis == -1:
            raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")

        print(f"\t\tSelect image with nuclei from IF image (channel index: {channel_id_for_registration})", flush=True)
        # # select nuclei channel from IF image
        # if channel_name_for_registration is None:
        #     raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")

        # select dapi channel for registration and convert to numpy array
        nuclei_img = np.take(image, channel_id_for_registration, channel_axis).compute()

    # Setup image registration objects - is important to load and scale the images.
    # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.

    # First: Setup the ImageRegistration object for the whole image (before deconvolution in histo images and multi-channel in IF)
    imreg_complete = ImageRegistration(
        image=image,
        template=template,
        axes_image=axes_image,
        axes_template=axes_template,
        verbose=True
        )
    # load and scale the whole image
    print('Load and scale image data containing all channels.')
    imreg_complete.load_and_scale_images()

    # setup ImageRegistration object with the nucleus image (either from deconvolution or just selected from IF image)
    axes_selected = "YX"
    imreg_selected = ImageRegistration(
        image=nuclei_img,
        template=imreg_complete.template,
        axes_image=axes_selected,
        axes_template=axes_template,
        max_width=4000,
        convert_to_grayscale=False,
        perspective_transform=False,
        min_good_matches=min_good_matches
    )

    # run all steps to extract features and get transformation matrix
    print('Load and scale image data containing only the channels required for registration.')
    imreg_selected.load_and_scale_images()

    print("\t\tExtract common features from image and template", flush=True)
    # perform registration to extract the common features ptsA and ptsB
    imreg_selected.extract_features(test_flipping=test_flipping)
    imreg_selected.calculate_transformation_matrix()

    if image_type == "histo":
        # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
        if imreg_complete.image_resized is None:
            imreg_selected.image = imreg_complete.image  # use original image
        else:
            imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image

        # perform registration
        imreg_selected.perform_registration()

        if save_registered_images:
            # save files
            identifier = f"{prefix}__{data.slide_id}__{data.sample_id}__{channel_names[0]}"
            imreg_selected.save(
                output_dir=output_dir,
                identifier = identifier,
                axes=axes_image,
                photometric='rgb',
                ome_metadata=ome_metadata
                )

            # # save metadata
            # data.metadata["method_params"]['images'][f'registered_{channel_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
            # write_dict_to_json(data.metadata["method_params"], data.path / "experiment_modified.xenium")
            # #self._save_metadata_after_registration()

        data.images.add_image(
            image=imreg_selected.registered,
            name=channel_names[0],
            axes=axes_image,
            pixel_size=pixel_size,
            ome_meta=ome_metadata,
            overwrite=True
            )

        del imreg_complete, imreg_selected, image, template, nuclei_img, eo, dab
    else:
        # image_type is IF
        # In case of IF images the channels are normally in the first axis and each channel is registered separately
        # Further, each channel is then saved separately as grayscale image.

        # iterate over channels
        for i, n in enumerate(channel_names):
            # skip the DAPI image
            if n == channel_name_for_registration:
                break

            if imreg_complete.image_resized is None:
                # select one channel from non-resized original image
                imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
            else:
                # select one channel from resized original image
                imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)

            # perform registration
            imreg_selected.perform_registration()

            if save_registered_images:
                # save files
                identifier = f"{data.slide_id}__{data.sample_id}__{n}"

                imreg_selected.save(
                    output_dir=output_dir,
                    identifier=identifier,
                    axes='YX',
                    photometric='minisblack',
                    ome_metadata=ome_metadata
                    )

                # # save metadata
                # data.metadata["method_params"]['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, data.path).replace("\\", "/")
                # write_dict_to_json(data.metadata["method_params"], data.path / "experiment_modified.xenium")
                # #self._save_metadata_after_registration()
            # if add_registered_image:
            data.images.add_image(
                image=imreg_selected.registered,
                name=n,
                axes=axes_image,
                pixel_size=pixel_size,
                ome_meta=ome_metadata,
                overwrite=True
                )

        # free RAM
        del imreg_complete, imreg_selected, image, template, nuclei_img
    gc.collect()