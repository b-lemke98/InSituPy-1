from pathlib import Path

# make sure that images do not exceed limits in c++ (required for cv2::remap function in cv2::warpAffine)
# see also https://www.geeksforgeeks.org/climits-limits-h-cc/
SHRT_MAX = 2**15-1 # 32767
SHRT_MIN = -(2**15-1) # -32767

# create cache dir
CACHE = Path.home() / ".cache/InSituPy/"

# naming
INSITUDATA_EXTENSION = ".ispy"
