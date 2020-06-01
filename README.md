# gan

The goal of this project was to implement a framework to allow a user to quickly experiment with different models and datasets to train Generation Adverserial Networks (GANs). The project uses a similar structure to my stvNet project, with model structures being handled in the `models.py` file, data-related functions being handled in the `data.py` file, and the main training function being handled in the `train.py` file.

To train a set of models, a user can call the `train` function in the `train.py` file, using any combination of desired generator and discriminator models contained in the `models.py` file, along with a compatible generator function which provides a dataset to train against, and several other hyperparameter variables.

The project also allows a user to quickly build a new dataset by using the `image_from_urls` function in the `data.py` file, which takes a specified text file containing urls to a set of desired images, and stores the images in the specified folder. These text files can be built by using the `http://image-net.org/` image database. I also plan to integrate the use of the `google_images_download ` library in the future to generate these files but the library was [not functioning](https://github.com/hardikvasa/google-images-download/pull/298) at the time the project was developed.
