from pathlib import Path
import os.path
from typing import TYPE_CHECKING, Literal
from insitupy.datasets.download import download_url
import shutil

import sys
import hashlib

#functions that each download a dataset into  '~/.cache/InSituPy/demo_dataset'

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
        over_write: bool = False
) -> None:
        
    if os.path.exists(os.path.expanduser("~") / Path('.cache/InSituPy/demo_dataset/output-XETG00000__slide_id__hbreastcancer')) and md5sum(os.path.expanduser("~") / Path('.cache/InSituPy/demo_dataset/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip'))=='7d42a0b232f92a2e51de1f513b1a44fd':
        if not over_write:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    path = os.path.expanduser("~") / Path('.cache/InSituPy')
    path.mkdir(parents=True,exist_ok=True)

    out_dir = path / Path('demo_dataset')
    image_dir = out_dir / 'unregistered_images'
    data_dir = out_dir / "output-XETG00000__slide_id__hbreastcancer"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif" 
    
    download_url(xeniumdata_url, out_dir = out_dir, overwrite = over_write)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__hbreastcancer__HE__histo", overwrite = over_write)
    download_url(if_url, out_dir=image_dir, file_name="slide_id__hbreastcancer__CD20_HER2_DAPI__IF", overwrite = over_write)

    zip_file = list(out_dir.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)
        
    os.rmdir(data_dir / "outs")

#spaceranger version 1.5.0
#data from https://www.10xgenomics.com/resources/datasets/human-kidney-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
def nondiseased_kidney(
        over_write: bool = False
) -> None:
    
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hkidney")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xenium_V1_hKidney_nondiseased_section_outs.zip"))=="194d5e21b40b27fa8c009d4cbdc3272d":
        if not over_write:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    path = os.path.expanduser("~") / Path('.cache/InSituPy')
    path.mkdir(parents=True,exist_ok=True)

    out_dir = path / Path('demo_dataset')
    image_dir = out_dir / 'unregistered_images'
    data_dir = out_dir / "output-XETG0000__slide_id__hkidney"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif"

    download_url(xeniumdata_url, out_dir = out_dir, overwrite = over_write)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__hkidney__HE__histo", overwrite = over_write)
    
    zip_file = list(out_dir.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)

#spaceranger version 1.6.0
#data from https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard 
def pancreatic_cancer(
        over_write: bool = False

) -> None:
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hPancreas")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"))=="7acca4c2a40f09968b72275403c29f93":
        if not over_write:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    path = os.path.expanduser("~") / Path('.cache/InSituPy')
    path.mkdir(parents=True,exist_ok=True)

    out_dir = path / Path('demo_dataset')
    image_dir = out_dir / 'unregistered_images'
    data_dir = out_dir / "output-XETG0000__slide_id__hPancreas"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif"
    if_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_if_image.ome.tif"

    download_url(xeniumdata_url, out_dir = out_dir, overwrite = over_write)
    download_url(he_url, out_dir = image_dir, file_name = "slide_id__hPancreas__HE__histo", overwrite = over_write)
    download_url(if_url, out_dir = image_dir, file_name = "slide_id__hPancreas__CD20_TROP2_PPY_DAPI__IF", overwrite = over_write )
    
    zip_file = list(out_dir.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)  

#spaceranger version 1.7.0    
#data from https://www.10xgenomics.com/resources/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard
def hskin_melanoma(
        over_write: bool = False
) -> None:
    if os.path.exists(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/output-XETG0000__slide_id__hskin")) and md5sum(os.path.expanduser("~") / Path(".cache/InSituPy/demo_dataset/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"))=="29102799a3f1858c7318b705eb1a8584":
        if not over_write:
            print(f"This Data exists already. Download is skipped. To force download set `over_write=True`.")
            return
        else:
            pass

    path = os.path.expanduser("~") / Path('.cache/InSituPy')
    path.mkdir(parents=True,exist_ok=True)

    out_dir = path / Path('demo_dataset')
    image_dir = out_dir / 'unregistered_images'
    data_dir = out_dir / "output-XETG0000__slide_id__hskin"

    xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
    he_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_he_image.ome.tif"

    download_url(xeniumdata_url, out_dir = out_dir, overwrite = over_write)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__hskin__HE__histo", overwrite = over_write)
    
    zip_file = list(out_dir.glob("*.zip"))[0]

    shutil.unpack_archive(zip_file, data_dir)
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)    


