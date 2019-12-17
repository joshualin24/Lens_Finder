# Lens_Finder



This is the source code for [strong lensing finding challenge 2.0](http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html). The dataformat is described by the website. We aim to use a neural network approach to classify whether a image (composed of 4 channels, 'EUC_H', 'EUC_J', 'EUC_Y', 'EUC_VIS') contain strong lens or not.

The original training set contain roughly 100,000 images. We split them into 80% of training and 20% of validating set. To fully utilize the data given, we use infomation from all the channels given. In order to do that, we change the input archeicture of neural network to 4 input channels. We use ResNet 18 as a baseline neural network at this moment.

