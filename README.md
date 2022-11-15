# Learning without real data, a 3D simulated data learning approach applied for robust ID cards text extraction. (SAC 2023)

This is a Python implementation of the following paper: Learning without real data, a 3D data simulation learning approach for robust ID cards segmentation and text extraction.
We are sharing a reusable public dataset of realistic synthetic French identity cards fully annotated in terms of textual content as well as information
position allowing other researchers to train their own models. We are also sharing the 15 models we trained.

## Datasets

You can download the differents datasets by clicking on the links:

* 5000 random Identity Cards generated on Blender : https://www.mediafire.com/file/8n6gg4uaics1ms9/5K_EVAL_QUANTITATIVE.zip/file
* 3000 Identity Cards generated on Blender using the Active Learning Uncertainty Sampling method : https://www.mediafire.com/file/72qe1a5fd54zl1b/AL_3K.zip/file

## Models

You can download the different models by clicking on the links:

* EB-UNet RS : https://www.mediafire.com/file/37jmnbk71uiwm70/model_ebunet_dataset_random_3k.pth/file
* EB-UNet AL : https://www.mediafire.com/file/5wduhva7io60774/model_ebunet_AL_3K.pth/file
* EB-UNet DA : https://www.mediafire.com/file/fnzglel4kt6fa9t/model_ebunet_DA_3K.pth/file
* EB-UNet RS+DA : https://www.mediafire.com/file/4dr1naq4a614u9g/model_ebunet_random_DA_3K.pth/file
* EB-UNet AL+DA : https://www.mediafire.com/file/95fw3655ui0mzz4/model_ebunet_ALDA_3K.pth/file
* UNet RS : https://www.mediafire.com/file/gloks7w3bb1qwk5/model_unet_random_3k.pth/file
* UNet AL : https://www.mediafire.com/file/ooqxl7ihylwf1hp/model_unet_AL_3k.pth/file
* UNet DA : https://www.mediafire.com/file/xlofhx7d62940gw/model_unet_DA_3k.pth/file
* UNet RS+DA : https://www.mediafire.com/file/zrmzekm7lhuj1e8/model_unet_hybrid_3k.pth/file
* UNet AL+DA : https://www.mediafire.com/file/95fw3655ui0mzz4/model_ebunet_ALDA_3K.pth/file
* KPR RS : https://www.mediafire.com/file/l6wxkzzejb81yoq/model_kpr_dataset_random_3k.pth/file
* KPR AL : https://www.mediafire.com/file/r00umtk485m3ylh/model_kpr_AL_3K.pth/file
* KPR DA : https://www.mediafire.com/file/uehz7aj1692iih9/model_kpr_DA_3K.pth/file
* KPR RS+DA : https://www.mediafire.com/file/fveks7odk1b8lwj/model_kpr_moitmoit3k.pth/file
* KPR AL+DA : https://www.mediafire.com/file/4h2mc78796pp68n/model_kpr_ALDA_3K.pth/file
