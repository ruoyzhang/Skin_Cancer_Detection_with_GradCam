# MelaNet

## Skin cancer classification

#### *Robin Ali - Louise Badarani - Cyriac Parisot - Clément Ponsonnet - Ruoy Zhang*

On average between 2 and 3 million skin cancers are diagnosed yearly world wide (World Health Organization). AI has been proven as a powerful diagnostic tool in medical fiels. We thus aim to develop a classifier to help dermathologist assess their diagnostics and understand the most prominent characteristics of each cancer types.

We trained and tested our models on the HAM10000 ("Human Against Machine with 10000 training images") dataset, a set of labeled dermatoscopic images from different populations, acquired and stored by different modalities.



7 types of skin cancers are being detected:
 - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
 - basal cell carcinoma (bcc)
 - benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
 - dermatofibroma (df)
 - melanoma (mel)
 - melanocytic nevi (nv)
 - vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

Our VGG16 model achieved 92% accuracy on our test set after 100 epochs on the classification task.

The GradCam was able to detect discriminating visual features of particular legion classes.
See the images below for illustration:

The Original Picture:

![Original Picture](https://github.com/ruoyzhang/Skin_Cancer_Detection_with_GradCam/blob/master/Sampled_images/ISIC_0029319.jpg)

The GradCam Overlay:

![GradCam Overlay](https://github.com/ruoyzhang/Skin_Cancer_Detection_with_GradCam/blob/master/Sampled_images/cam_ISIC_0029319.jpg)

The detailed presentation is in [presentation.ipynb](presentation.ipynb).


