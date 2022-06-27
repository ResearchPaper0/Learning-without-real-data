# Learning without real data, a 3D simulated data learning approach applied for robust ID cards text extraction. (PRICAI 2022)

This is a Python implementation of the following paper: Learning without real data, a 3D simulated data learning approach applied for robust ID cards text extraction.
We are sharing a reusable public dataset of realistic synthetic French identity cards fully annotated in terms of textual content as well as information
position allowing other researchers to train their own models. We are also sharing the 8 models we trained.

## Datasets

You can download the differents datasets by clicking on the links:

* 5000 random Identity Cards generated on Blender : https://www.mediafire.com/file/8n6gg4uaics1ms9/5K_EVAL_QUANTITATIVE.zip/file
* 3000 Identity Cards generated on Blender using the Active Learning Uncertainty Sampling method : https://www.mediafire.com/file/72qe1a5fd54zl1b/AL_3K.zip/file

## Models

You can download the different models by clicking on the links:

* EB-Unet    : https://www.mediafire.com/file/37jmnbk71uiwm70/model_ebunet_dataset_random_3k.pth/file
* EB-Unet AL : https://www.mediafire.com/file/5wduhva7io60774/model_ebunet_AL_3K.pth/file
* EB-Unet DA : https://www.mediafire.com/file/fnzglel4kt6fa9t/model_ebunet_DA_3K.pth/file
* EB-Unet Hybrid : https://www.mediafire.com/file/4dr1naq4a614u9g/model_ebunet_random_DA_3K.pth/file
* KPR : https://www.mediafire.com/file/l6wxkzzejb81yoq/model_kpr_dataset_random_3k.pth/file
* KPR AL : https://www.mediafire.com/file/r00umtk485m3ylh/model_kpr_AL_3K.pth/file
* KPR DA : https://www.mediafire.com/file/uehz7aj1692iih9/model_kpr_DA_3K.pth/file
* KPR Hybrid : https://www.mediafire.com/file/fveks7odk1b8lwj/model_kpr_moitmoit3k.pth/file
