import numpy as np
import cv2
from datetime import datetime
import numpy as np
from .manipulation import resize_image, convert_to_8bit
import dask.array as da
from typing import Optional, Tuple, Union, List, Dict, Any, Literal

# limits in C (see https://www.geeksforgeeks.org/climits-limits-h-cc/)
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # 32767

def register_image(image: Union[np.ndarray, da.Array],
                   template: Union[np.ndarray, da.Array], 
                   maxFeatures: int = 500, 
                   keepFraction=0.2, 
                   maxpx: Optional[int] = None,
                   method: Literal["sift", "surf"] = "sift", 
                   ratio_test: bool = True, 
                   flann: bool = True, 
                   perspective_transform: bool = False, 
                   do_registration: bool = True,
                   return_grayscale: bool = True,
                   return_features: bool = False,
                   verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    verboseprint = print if verbose else lambda *a, **k: None
    
    # load images into memory if they are dask arrays
    if isinstance(image, da.Array):
        verboseprint("Load image into memory...", flush=True)
        image = image.compute() # load into memory
    
    if isinstance(template, da.Array):
        verboseprint("Load template into memory...", flush=True)
        template = template.compute() # load into memory

    # check format
    if len(image.shape) == 3:
        verboseprint("Convert image to grayscale...")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(template.shape) == 3:
        verboseprint("Convert template to grayscale...")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # scale_factor = 0.2
    # if scale_factor < 1:
    #     print("Scale images before registration by factor {}".format(scale_factor))
    #     image_scaled = resize_image(img=image, scale_factor=scale_factor)
    #     template_scaled = resize_image(img=template, scale_factor=scale_factor)
    # else:
    #     image_scaled = image
    #     template_scaled = template

    # dim = (4000,4000)
    if maxpx is not None:
        if np.max(image.shape) > maxpx:
            shape_image = tuple([int(elem / np.max(image.shape) * maxpx) for elem in image.shape])
        else:
            shape_image = image.shape
        
        if np.max(template.shape) > maxpx:        
            shape_template = tuple([int(elem / np.max(template.shape) * maxpx) for elem in template.shape])
        else:
            shape_template = template.shape
            
        # reverse order of shape to match opencv requirements
        dim_image = (shape_image[1], shape_image[0])
        dim_template = (shape_template[1], shape_template[0])
            
        verboseprint("Rescale image to following dimensions: {}".format(shape_image))
        verboseprint("Rescale template to following dimensions: {}".format(shape_template))
        image_scaled = resize_image(img=image, dim=dim_image)
        template_scaled = resize_image(img=template, dim=dim_template)
        verboseprint("Dim of image: {}".format(image_scaled.shape))
        verboseprint("Dim of template: {}".format(template_scaled.shape))
    else:
        image_scaled = image
        template_scaled = template

    # convert and normalize images to 8bit for registration
    verboseprint("Convert scaled images to 8 bit")
    image_scaled = convert_to_8bit(image_scaled)
    template_scaled = convert_to_8bit(template_scaled)

    verboseprint("{}: Get features...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
    # Get features
    if method == "sift":
        verboseprint("     Method: SIFT...")
        # sift
        sift = cv2.SIFT_create()

        (kpsA, descsA) = sift.detectAndCompute(image_scaled, None)
        (kpsB, descsB) = sift.detectAndCompute(template_scaled, None)

    elif method == "surf":
        verboseprint("     Method: SURF...")
        surf = cv2.xfeatures2d.SURF_create(400)

        (kpsA, descsA) = surf.detectAndCompute(image_scaled, None)
        (kpsB, descsB) = surf.detectAndCompute(template_scaled, None)

    else:
        verboseprint("Unknown method. Aborted.")
        return

    if flann:
        verboseprint("{}: Compute matches...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        # runn Flann matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descsA, descsB, k=2)

    else:
        verboseprint("{}: Compute matches...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # feature matching
        #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descsA, descsB, k=2)

    if ratio_test:
        verboseprint("{}: Filter matches...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)
    else:
        verboseprint("{}: Filter matches...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches = sorted(matches, key=lambda x: x.distance)
        # keep only the top matches
        keep = int(len(matches) * keepFraction)
        good_matches = matches[:keep]

        verboseprint("Number of matches used: {}".format(len(good_matches)))

    # visualize the matched keypoints
    verboseprint("{}: Display matches...".format(f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
    matchedVis = cv2.drawMatches(image_scaled, kpsA, template_scaled, kpsB, good_matches, None)
    
    # Get keypoints
    verboseprint("{}: Fetch keypoints...".format(
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
    # allocate memory for the keypoints (x, y)-coordinates of the top matches
    ptsA = np.zeros((len(good_matches), 2), dtype="float")
    ptsB = np.zeros((len(good_matches), 2), dtype="float")
    # loop over the top matches
    for (i, m) in enumerate(good_matches):
        # indicate that the two keypoints in the respective images map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # calculate scale factors for x and y dimension for image and template
    x_sf_image = shape_image[0] / image.shape[0]
    y_sf_image = shape_image[1] / image.shape[1]
    x_sf_template = shape_template[0] / template.shape[0]
    y_sf_template = shape_template[1] / template.shape[1]

    # apply scale factors to points - separately for each dimension
    ptsA[:, 0] = ptsA[:, 0] / x_sf_image
    ptsA[:, 1] = ptsA[:, 1] / y_sf_image
    ptsB[:, 0] = ptsB[:, 0] / x_sf_template
    ptsB[:, 1] = ptsB[:, 1] / y_sf_template

    # # apply scale_factor to points
    # ptsA /= scale_factor
    # ptsB /= scale_factor

    if perspective_transform:
        # compute the homography matrix between the two sets of matched
        # points
        verboseprint("{}: Compute homography matrix...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    else:
        verboseprint("{}: Estimate 2D affine transformation matrix...".format(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
        (H, mask) = cv2.estimateAffine2D(ptsA, ptsB)

    # use the homography matrix to register the images
    (h, w) = template.shape[:2]
    if do_registration:
        if perspective_transform:
            # warping
            verboseprint("{}: Register image by perspective transformation...".format(
                f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
            
            image = convert_to_8bit(image)
            registered = cv2.warpPerspective(image, H, (w, h))
        else:
            verboseprint("{}: Register image by affine transformation...".format(
                f"{datetime.now():%Y-%m-%d %H:%M:%S}"))
            
            image = convert_to_8bit(image)
            registered = cv2.warpAffine(image, H, (w, h))

        if return_grayscale:
            if len(registered.shape) == 3:
                verboseprint("Convert registered image to grayscale...")
                registered = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    else:
        registered = None

    # return the registered image
    if return_features:
        return registered, H, matchedVis, ptsA, ptsB
    else:
        return registered, H, matchedVis

