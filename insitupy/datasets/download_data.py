from pathlib import Path
import os.path
from typing import TYPE_CHECKING, Literal
from insitupy.datasets.download import download_url
import shutil
import hashlib

#Data from:
#https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
#https://www.10xgenomics.com/resources/datasets/human-kidney-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard
#https://www.10xgenomics.com/resources/datasets/human-skin-preview-data-xenium-human-skin-gene-expression-panel-add-on-1-standard

#code adapted from scanpy/datasets
#for now 3 datasets included, choose by Literal:
# "Xenium_FFPE_Human_Breast_Cancer_Rep1",
#"V1_hKidney_nondiseased_section",
#"V1_hSkin_Melanoma_Add_on_FFPE"

def dataload(
        sample_id: Literal[
        "Xenium_FFPE_Human_Breast_Cancer_Rep1",
        "V1_hKidney_nondiseased_section",
        "V1_hSkin_Melanoma_Add_on_FFPE",
        #add more
        ],
):
    
    over_write: bool = False
    #check, if demo_dataset already exists, if yes overwrite
    if os.path.exists("demo_dataset"): 
        over_write = True
        shutil.rmtree('demo_dataset')
        print('demo_dataset already exists, overwrite with new dataset')
    
    out_dir = Path("demo_dataset") # output directory
    image_dir = out_dir / "unregistered_images" # directory for images
    if_url = None
  
    if  sample_id == "Xenium_FFPE_Human_Breast_Cancer_Rep1":
        xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
        he_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
        if_url = "https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"  

    if sample_id == "V1_hKidney_nondiseased_section":
        xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_outs.zip"
        he_url = "https://cf.10xgenomics.com/samples/xenium/1.5.0/Xenium_V1_hKidney_nondiseased_section/Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif"
    if sample_id == "V1_hSkin_Melanoma_Add_on_FFPE":
        xeniumdata_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip"
        he_url = "https://cf.10xgenomics.com/samples/xenium/1.7.0/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_he_image.ome.tif"

    #Download data from url

    download_url(xeniumdata_url, out_dir = out_dir, overwrite = over_write)
    download_url(he_url, out_dir = image_dir, file_name="slide_id__sample_id__HE__histo",overwrite = over_write)

    if if_url is not None:
        download_url(if_url, out_dir=image_dir, file_name="slide_id__sample_id__CD20_HER2_DAPI__IF",overwrite = over_write)

    zip_file = list(out_dir.glob("*.zip"))[0]
    data_dir = out_dir / "output-XETG00000__slide_id__sample_id"
    
    shutil.unpack_archive(zip_file, data_dir)

    # move output files to right position
    for f in data_dir.glob("outs/*"):
        shutil.move(f, data_dir)
        
    # remove empty outs folder
    os.rmdir(data_dir / "outs")
    
#dataload('Xenium_FFPE_Human_Breast_Cancer_Rep1')
#dataload('V1_hKidney_nondiseased_section')
#dataload("V1_hSkin_Melanoma_Add_on_FFPE")
