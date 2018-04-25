# Scan-path Prediction on 360◦ Images using Saliency Volumes

| ![Marc Assens][MarcAssens-photo] | ![Kevin McGuinness][KevinMcGuinness-photo] | ![Xavier Giro-i-Nieto][XavierGiro-photo]| ![Noel O'Connor][NoelOConnor-photo] |
|:-:|:-:|:-:|:-:|
| [Marc Assens][MarcAssens-web]  | [Kevin McGuinness][KevinMcGuinness-web]  | [Xavier Giro-i-Nieto][XavierGiro-web] | [Noel O'Connor][NoelOConnor-web]   |

[MarcAssens-web]: https://www.linkedin.com/in/marc-assens-reina-5b1090bb/
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[MarcAssens-photo]: https://github.com/massens/saliency-360salient-2017/raw/master/authors/foto_carnet_dublin.jpg "Marc Assens"
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/Kevin160x160%202.jpg?token=AFOjyZmLlX3ZgpkNe60Vn3ruTsq01rD9ks5YdAaiwA%3D%3D "Kevin McGuinness"
[XavierGiro-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/NoelOConnor.jpg "Noel O'Connor"


A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-gpi] |
|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web] | [UPC Image Processing Group][gpi-web] |

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"


## Abstract

We introduce a deep neural network for scan-path predic- tion trained on 360◦ images, and a temporal-aware novel representation of saliency information named saliency vol- ume. The first stage of the network consits of a model trained to generate saliency volumes, whose weights are learned by back-propagation computed from a binary cross en- tropy (BCE) loss over downsampled versions of the saliency volumes. Sampling strategies are used to generate scan- paths from saliency volumes. Our experiments show the advantages of using saliency volumes, and how they can be used for related tasks. 

## Publication

Find the pre-print version of our work on [arXiv](https://arxiv.org/abs/1707.03123).

![Image of the paper](https://github.com/massens/saliency-360salient-2017/raw/master/figs/paper.png)

Please cite with the following Bibtex code:

```
@InProceedings{Reina_2017_ICCV_Workshops,
author = {Assens Reina, Marc and Giró-i-Nieto, Xavier and McGuinness, Kevin and O'Connor, Noel E.},
title = {SaltiNet: Scan-Path Prediction on 360 Degree Images Using Saliency Volumes},
booktitle = {ICCV Workshop on Egocentric Perception, Interaction and Computing},
month = {Oct},
year = {2017}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Marc Assens, Kevin McGuinness, Xavier Giro-i-Nieto and Noel E. O’Connor. “SaltiNet: Scan-Path Prediction on 360 Degree Images Using Saliency Volumes.” ICCV Workshop on Egocentric Perception, Interaction and Computing. 2017.*


## Slides

<center>
<iframe src="//www.slideshare.net/slideshow/embed_code/key/crpL3byLoainOX" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/saltinet-the-temporal-dimension-of-visual-attention-models" title="SaltiNet: The Temporal Dimension of Visual Attention Models " target="_blank">SaltiNet: The Temporal Dimension of Visual Attention Models </a> </strong> from <strong><a target="_blank" href="https://www.slideshare.net/xavigiro">Xavier Giro-i-Nieto</a></strong> </div>
</center>

## Models

The scan-path generator presented in our work can be downloaded from the links provided below the figure:

Model Architecture:
![architecture-fig]


* [[Scan-path generator model (100 MB)]](https://github.com/massens/saliency-360salient-2017/blob/master/src/pathnet_model_v2.h5)

[architecture-fig]: https://github.com/massens/saliency-360salient-2017/raw/master/figs/model.png "Model architecture"

## Saliency volumes
Saliency volumes aim to be a suitable representaiton of spatial and temporal saliency information for images. They have three axis that represent width and height of the image, and time. They are a meta representation of saliency information and other saliency representations can be extracted from them. Saliency maps can be generated performing an addition operation of all the temporal slices of the volume, and normalizing the values to ensure they add to one. A simillar representaiton are temporally wheigted saliency maps, which are generated by performing a weighted addition operation of all the temporal slices. Finally, scan-paths can also be extracted by sampling fixation points from the temporal slices. 

![salvol-fig]

[salvol-fig]: https://github.com/massens/saliency-360salient-2017/raw/master/figs/salvol.png "Saliency Volume"
## Datasets

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [360 Salient Challenge](http://www.icme2017.org/grand-challenges/). Notice that, after our publication, this dataset has been updated and published in [this site](https://salient360.ls2n.fr/datasets/).

## Software frameworks: Keras

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
```
pip install -r https://github.com/massens/saliency-360salient-2017/blob/master/requirements.txt
```

## Acknowledgements

We would like to especially thank Albert Gil Moreno from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the projects [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application) and [Malegra TEC2016-75976-R](https://imatge.upc.edu/web/projects/malegra-multimodal-signal-processing-and-machine-learning-graphs), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289. |  ![logo-ireland] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"
[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/saliency-salgan-2017/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.
pat
