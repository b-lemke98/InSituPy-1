import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from .._constants import SHRT_MAX, SHRT_MIN
from .._exceptions import NotEnoughFeatureMatchesError
from ..utils.utils import remove_last_line_from_csv
from .io import write_ome_tiff
from .utils import convert_to_8bit, fit_image_to_size_limit, scale_to_max_width


class ImageRegistration:
    '''
    Object to perform image registration.
    '''
    def __init__(self,
                 image: Union[np.ndarray, da.Array],
                 template: Union[np.ndarray, da.Array],
                 axes_image: str = "YXS", ## channel axes - other examples: 'TCYXS'. S for RGB channels.
                 axes_template: str = "YX",  # channel axes of template. Normally it is just a grayscale image - therefore YX.
                 max_width: Optional[int] = 4000,
                 convert_to_grayscale: bool = False,
                 perspective_transform: bool = False,
                 feature_detection_method: Literal["sift", "surf"] = "sift",
                 flann: bool = True,
                 ratio_test: bool = True,
                 keepFraction: float = 0.2,
                 min_good_matches: int = 20,  # minimum number of good feature matches
                 maxFeatures: int = 500,
                 verbose: bool = True,
                 ):

        # check verbose mode
        self.verboseprint = print if verbose else lambda *a, **k: None

        # add arguments to object
        self.image = image
        self.template = template
        self.axes_image = axes_image
        self.axes_template = axes_template
        self.max_width = max_width
        self.convert_to_grayscale = convert_to_grayscale
        self.perspective_transform = perspective_transform
        self.feature_detection_method = feature_detection_method
        self.flann = flann
        self.ratio_test = ratio_test
        self.keepFraction = keepFraction
        self.min_good_matches = min_good_matches
        self.maxFeatures = maxFeatures
        self.verbose = verbose

    def load_and_scale_images(self):

        # load images into memory if they are dask arrays
        if isinstance(self.image, da.Array):
            self.verboseprint("\t\tLoad image into memory...", flush=True)
            self.image = self.image.compute()  # load into memory

        if isinstance(self.template, da.Array):
            self.verboseprint("\t\tLoad template into memory...", flush=True)
            self.template = self.template.compute()  # load into memory

        if self.convert_to_grayscale:
            # check format
            if len(self.image.shape) == 3:
                self.verboseprint("\t\tConvert image to grayscale...")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if len(self.template.shape) == 3:
                self.verboseprint("\t\tConvert template to grayscale...")
                self.template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        if self.max_width is not None:
            self.verboseprint("\t\tRescale image and template to save memory.", flush=True)
            self.image_scaled = scale_to_max_width(self.image,
                                                   axes=self.axes_image,
                                                   max_width=self.max_width,
                                                   use_square_area=True,
                                                   verbose=self.verbose,
                                                   print_spacer="\t\t\t"
                                                   )
            self.template_scaled = scale_to_max_width(self.template,
                                                      axes=self.axes_template,
                                                      max_width=self.max_width,
                                                      use_square_area=True,
                                                      verbose=self.verbose,
                                                      print_spacer="\t\t\t"
                                                      )
        else:
            self.image_scaled = self.image
            self.template_scaled = self.template

        # convert and normalize images to 8bit for registration
        self.verboseprint("\t\tConvert scaled images to 8 bit")
        self.image_scaled = convert_to_8bit(self.image_scaled)
        self.template_scaled = convert_to_8bit(self.template_scaled)

        # calculate scale factors for x and y dimension for image and template
        # TODO: Do we really nead to do this separately for both axes?
        self.x_sf_image = self.image_scaled.shape[1] / self.image.shape[1]
        self.y_sf_image = self.image_scaled.shape[0] / self.image.shape[0]
        self.x_sf_template = self.template_scaled.shape[1] / self.template.shape[1]
        self.y_sf_template = self.template_scaled.shape[0] / self.template.shape[0]

        # resize image if necessary (warpAffine has a size limit for the image that is transformed)
        if np.any([elem > SHRT_MAX for elem in self.image.shape[:2]]):
            self.verboseprint(
                "\t\tWarning: Dimensions of image ({}) exceed C++ limit SHRT_MAX ({}). " \
                "Image dimensions are resized to meet requirements. " \
                "This leads to a loss of quality.".format(self.image.shape, SHRT_MAX))

            # fit image
            self.image_resized, self.resize_factor_image = fit_image_to_size_limit(
                self.image, size_limit=SHRT_MAX, return_scale_factor=True, axes=self.axes_image
                )
        else:
            self.image_resized = None
            self.resize_factor_image = 1

    def extract_features(self):
        '''
        Function to extract paired features from image and template.
        '''

        self.verboseprint("\t\t{}: Get features...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))

        # Test different flip transformations starting with no flip, then vertical, then horizontal.
        matches_list = [] # list to collect number of matches
        for flip_axis in [None, 0, 1]:
            flipped = False
            if flip_axis is not None:
                # flip image
                print(f"\t\t{'Vertical' if flip_axis == 0 else 'Horizontal'} flip is tested.", flush=True)
                self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)
                flipped = True # set flipped flag to True

            # Get features
            if self.feature_detection_method == "sift":
                self.verboseprint("\t\t\tMethod: SIFT...")
                # sift
                sift = cv2.SIFT_create()

                (kpsA, descsA) = sift.detectAndCompute(self.image_scaled, None)
                (kpsB, descsB) = sift.detectAndCompute(self.template_scaled, None)

            elif self.feature_detection_method == "surf":
                self.verboseprint("\t\t\tMethod: SURF...")
                surf = cv2.xfeatures2d.SURF_create(400)

                (kpsA, descsA) = surf.detectAndCompute(self.image_scaled, None)
                (kpsB, descsB) = surf.detectAndCompute(self.template_scaled, None)

            else:
                self.verboseprint("\t\t\tUnknown method. Aborted.")
                return

            if self.flann:
                self.verboseprint("\t\t{}: Compute matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)   # or pass empty dictionary

                # runn Flann matcher
                fl = cv2.FlannBasedMatcher(index_params, search_params)
                matches = fl.knnMatch(descsA, descsB, k=2)

            else:
                self.verboseprint("\t\t{}: Compute matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # feature matching
                #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descsA, descsB, k=2)

            if self.ratio_test:
                self.verboseprint("\t\t{}: Filter matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # store all the good matches as per Lowe's ratio test.
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7*n.distance:
                        good_matches.append(m)
            else:
                self.verboseprint("\t\t{}: Filter matches...".format(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
                # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
                matches = sorted(matches, key=lambda x: x.distance)
                # keep only the top matches
                keep = int(len(matches) * self.keepFraction)
                good_matches = matches[:keep][:self.maxFeatures]

                self.verboseprint("\t\t\tNumber of matches used: {}".format(len(good_matches)))

            # check if a sufficient number of good matches was found
            matches_list.append(len(good_matches))
            if len(good_matches) >= self.min_good_matches:
                print(f"\t\t\tSufficient number of good matches found ({len(good_matches)}).")
                self.flip_axis = flip_axis
                break
            else:
                print(f"\t\t\tNumber of good matches ({len(good_matches)}) below threshold ({self.min_good_matches}). Flipping is tested.")
                if flipped:
                    # flip back
                    print("Flip back.", flush=True)
                    self.image_scaled = np.flip(self.image_scaled, axis=flip_axis)

        if not hasattr(self, "flip_axis"):
            raise NotEnoughFeatureMatchesError(number=np.max(matches_list), threshold=self.min_good_matches)

        # check to see if we should visualize the matched keypoints
        self.verboseprint("\t\t{}: Display matches...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        self.matchedVis = cv2.drawMatches(self.image_scaled, kpsA, self.template_scaled, kpsB,
                                        good_matches, None)

        # Get keypoints
        self.verboseprint("\t\t{}: Fetch keypoints...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # allocate memory for the keypoints (x, y)-coordinates of the top matches
        self.ptsA = np.zeros((len(good_matches), 2), dtype="float")
        self.ptsB = np.zeros((len(good_matches), 2), dtype="float")
        # loop over the top matches
        for (i, m) in enumerate(good_matches):
            # indicate that the two keypoints in the respective images map to each other
            self.ptsA[i] = kpsA[m.queryIdx].pt
            self.ptsB[i] = kpsB[m.trainIdx].pt

        # apply scale factors to points - separately for each dimension
        self.ptsA[:, 0] = self.ptsA[:, 0] / self.x_sf_image
        self.ptsA[:, 1] = self.ptsA[:, 1] / self.y_sf_image
        self.ptsB[:, 0] = self.ptsB[:, 0] / self.x_sf_template
        self.ptsB[:, 1] = self.ptsB[:, 1] / self.y_sf_template

    def calculate_transformation_matrix(self):
        '''
        Function to calculate the transformation matrix.
        '''

        if self.perspective_transform:
            # compute the homography matrix between the two sets of matched
            # points
            self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Compute homography matrix...")
            (self.T, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
        else:
            self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Estimate 2D affine transformation matrix...")
            (self.T, mask) = cv2.estimateAffine2D(self.ptsA, self.ptsB)

        if self.resize_factor_image != 1:
            if self.perspective_transform:
                self.verboseprint("\t\tEstimate perspective transformation matrix for resized image", flush=True)
                self.ptsA *= self.resize_factor_image # scale images features in case it was originally larger than the warpAffine limits
                (self.T_resized, mask) = cv2.findHomography(self.ptsA, self.ptsB, method=cv2.RANSAC)
            else:
                self.verboseprint("\t\tEstimate affine transformation matrix for resized image", flush=True)
                self.ptsA *= self.resize_factor_image # scale images features in case it was originally larger than the warpAffine limits
                (self.T_resized, mask) = cv2.estimateAffine2D(self.ptsA, self.ptsB)

    def perform_registration(self):

        # determine which image to be registered here
        if self.image_resized is None:
            self.image_to_register = self.image
            self.T_to_register = self.T
        else:
            self.image_to_register = self.image_resized
            self.T_to_register = self.T_resized

        # determine the kind of transformation
        warp_func, warp_name = (cv2.warpPerspective, "perspective") if self.perspective_transform else (cv2.warpAffine, "affine")

        if self.flip_axis is not None:
            print(f"\t\tImage is flipped {'vertically' if self.flip_axis == 0 else 'horizontally'}", flush=True)
            self.image_to_register = np.flip(self.image_to_register, axis=self.flip_axis)

        # use the transformation matrix to register the images
        (h, w) = self.template.shape[:2]
        # warping
        self.verboseprint(f"\t\t{datetime.now():%Y-%m-%d %H:%M:%S}: Register image by {warp_name} transformation...")

        #self.image_resized = convert_to_8bit(self.image_resized)
        self.registered = warp_func(self.image_to_register, self.T_to_register, (w, h))

    def register_images(self):
        '''
        Function running the registration including following steps:
            1. Loading of images
            2. Feature extraction
            3. Calculation of transformation matrix
            4. Registration of images based on transformation matrix
        '''
        # load and scale images
        self.load_and_scale_images()

        # run feature extraction
        self.extract_features()

        # calculate transformation matrix
        self.calculate_transformation_matrix()

        # perform registration
        self.perform_registration()

    def save(self,
             path: Union[str, os.PathLike, Path],
             filename: str,
             axes: str,  # string describing the channel axes, e.g. YXS or CYX
             photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb', # before I had rgb here. Xenium doc says minisblack
             ome_metadata: dict = {},
             registered: Optional[np.ndarray] = None,  # registered image
             _T: Optional[np.ndarray] = None,  # transformation matrix
             matchedVis: Optional[np.ndarray] = None  # image showing the matched visualization
             ):
        # Optionally the registered image, transformation matrix and matchedVis can be added externally.
        # Otherwise they are retrieved from self.
        if registered is None:
            registered = self.registered

        if _T is None:
            if self.resize_factor_image == 1:
                # if the image was not resized the transformation matrix to save is identical to the one used for registration
                T_to_save = self.T_to_register
            else:
                # if the image WAS resized the transformation matrix to save is not identical to the one used for registration
                # instead the transformation matrix before resizing needs to be used
                T_to_save = self.T

        if matchedVis is None:
            matchedVis = self.matchedVis

        # save registered image as OME-TIFF
        regimg_dir = path.parent / "registered_images"
        regimg_dir.mkdir(parents=True, exist_ok=True) # create folder for registered images
        self.outfile = regimg_dir / f"{filename}__registered.ome.tif"
        print(f"\t\tSave OME-TIFF to {self.outfile}", flush=True)
        write_ome_tiff(
            file=self.outfile,
            image=registered,
            axes=axes,
            photometric=photometric,
            overwrite=True,
            metadata=ome_metadata
            )

        # save registration QC files
        reg_dir = path.parent / "registration_qc"
        reg_dir.mkdir(parents=True, exist_ok=True) # create folder for QC outputs
        print(f"\t\tSave QC files to {reg_dir}", flush=True)

        # save transformation matrix
        T_to_save = np.vstack([T_to_save, [0,0,1]]) # add last line of affine transformation matrix
        T_csv = reg_dir / f"{filename}__T.csv"
        np.savetxt(T_csv, T_to_save, delimiter=",") # save as .csv file

        # remove last line break from csv since this gives error when importing to Xenium Explorer
        remove_last_line_from_csv(T_csv)

        # save image showing the number of key points found in both images during registration
        matchedVis_file = reg_dir / f"{filename}__common_features.pdf"
        plt.imshow(matchedVis)
        plt.savefig(matchedVis_file, dpi=400)
        plt.close()
