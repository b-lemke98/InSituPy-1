import hashlib
import os.path
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from insitupy.datasets.download import download_url

from .._constants import CACHE

# parameters for download functions
DEMODIR = CACHE / 'demo_datasets'

# functions that each download a dataset into  '~/.cache/InSituPy/demo_dataset'
def md5sum(filePath):
    with open(filePath, 'rb') as fh:
        m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()

#spaceranger version 1.0.1
#data from https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
def human_breast_cancer(
        overwrite: bool = False
) -> None:
    
    # URLs for download
    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"
    
    # set up paths
    named_data_dir = DEMODIR / "hbreastcancer"
    xeniumdata_dir = named_data_dir / "output-XETG00000__slide_id__hbreastcancer"
    image_dir = named_data_dir / "unregistered_images"
    zip_file = named_data_dir / Path(xeniumdata_url).name
    
    # check if file exists and has correct md5sum
    expected_md5sum = '7d42a0b232f92a2e51de1f513b1a44fd'
    
    # check if the unzipped xenium data exists
    download_xeniumdata = False
    if xeniumdata_dir.exists():
        # if it exists, everything is fine and it is assumed that the dataset was downloaded correctly. Overwrite is still checked.
        if not overwrite:
            print(f"This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
            return
        else:
            print(f"This dataset exists already but is overwritten because of `overwrite=True`.")
            download_xeniumdata = True
    else:
        # if unzipped xenium data does not exist, we need to check if a zip file exists, and if yes if its md5sum is correct    
        if zip_file.exists():
            print("ZIP file exists. Checking md5sum...")
            if md5sum(zip_file) == expected_md5sum:
                if not overwrite:
                    print(f"This dataset exists already. Download is skipped. To force download set `overwrite=True`.")
                    return
                else:
                    # if the md5sum matches but overwrite=True, the data is still downloaded
                    download_xeniumdata = True
            else:
                print("The dataset exists already but the md5sum is not as expected. Dataset is downloaded again.")
                download_xeniumdata = True
        else:
            download_xeniumdata = True
            
    if download_xeniumdata:
        # download xenium data as zip file
        download_url(xeniumdata_url, out_dir=named_data_dir, overwrite=True)
        
        # unzip xenium data
        shutil.unpack_archive(zip_file, xeniumdata_dir)
        
        # move files from outs folder into parent directory and remove the outs folder
        for f in xeniumdata_dir.glob("outs/*"):
            shutil.move(f, xeniumdata_dir)
        os.rmdir(xeniumdata_dir / "outs")
    
    # download image data
    #TODO: It would be great if also the image files are checked for their md5sum
    download_url(he_url, out_dir=image_dir, file_name="slide_id__hbreastcancer__HE__histo", overwrite = overwrite)
    download_url(if_url, out_dir=image_dir, file_name="slide_id__hbreastcancer__CD20_HER2_DAPI__IF", overwrite = overwrite)



#spaceranger version 1.5.0
#data from https://www.10xgenomics.com/resources/datasets/human-kidney-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
def nondiseased_kidney(
        overwrite: bool = False
) -> None:
    
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hkidney")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xenium_V1_hKidney_nondiseased_section_outs.zip"))=="194d5e21b40b27fa8c009d4cbdc3272d":
        if not overwrite:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    CACHE.mkdir(parents=True,exist_ok=True)

    data_dir = DEMODIR / "output-XETG0000__slide_id__hkidney"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif"

    download_url(xeniumdata_url, out_dir = DEMODIR, overwrite = overwrite)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__hkidney__HE__histo", overwrite = overwrite)
    
    zip_file = list(DEMODIR.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)

#spaceranger version 1.6.0
#data from https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard 
def pancreatic_cancer(
        overwrite: bool = False

) -> None:
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hPancreas")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"))=="7acca4c2a40f09968b72275403c29f93":
        if not overwrite:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    CACHE.mkdir(parents=True,exist_ok=True)

    data_dir = DEMODIR / "output-XETG0000__slide_id__hPancreas"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_if_image.ome.tif"

    download_url(xeniumdata_url, out_dir = DEMODIR, overwrite = overwrite)
    download_url(he_url, out_dir = image_dir, file_name = "slide_id__hPancreas__HE__histo", overwrite = overwrite)
    download_url(if_url, out_dir = image_dir, file_name = "slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF", overwrite = overwrite )
    
    zip_file = list(DEMODIR.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)  

#spaceranger version 1.7.0    
#data from https://www.10xgenomics.com/resources/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard
def hskin_melanoma(
        overwrite: bool = False
) -> None:
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hskin")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"))=="29102799a3f1858c7318b705eb1a8584":
        if not overwrite:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    CACHE.mkdir(parents=True,exist_ok=True)

    data_dir = DEMODIR / "output-XETG0000__slide_id__hskin"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_he_image.ome.tif"

    download_url(xeniumdata_url, out_dir = DEMODIR, overwrite = overwrite)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__hskin__HE__histo", overwrite = overwrite)
    
    zip_file = list(DEMODIR.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)    


