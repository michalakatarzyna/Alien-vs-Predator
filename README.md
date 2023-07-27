
## Overview

![pobrane](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/28bc0edd-c2b4-4bd9-8f7a-a1fd3cf7fa6e)


This project is an image classification model that can distinguish between images of Predators and Aliens. The model was trained on the Alien vs. Predator dataset using different data augmentations and architectures.


## Data Source

The dataset used for training and evaluation is obtained from Kaggle:kaggle.com/datasets/pmigdal/alien-vs-predator-images. It consists of images of Predators and Aliens, which are split into training and validation sets.

## Data Preprocessing



The images were preprocessed before being fed into the model. They were resized to a target size of 224x224 pixels and normalized to have values between 0 and 1.


![2](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/5d1b88a4-7189-463f-aff5-67c68078377e)
![1](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/815d13c0-bfa5-4e5a-974b-7d4c0131365a)



## Models
The project explores different models with various data augmentations:

Grayscale Model: A simple CNN model with grayscale images.


![3](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/66203599-fd93-4dd6-a946-28b6315c5aed)




RGB Model: A CNN model with RGB images.


![4](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/74562f07-9a89-4807-ad90-7d63a2018008)




Rotate Model: A CNN model with rotated images as data augmentation.


![5](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/436491a2-14c1-4707-a3e4-21cf7bb5f071)




Flip Model: A CNN model with horizontally and vertically flipped images as data augmentation.


![6](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/ee6f6b84-0920-4636-bd02-97513bdb4e23)



Brightness Model: A CNN model with brightness-adjusted images as data augmentation.


![7](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/4304ad4e-c91c-485f-a821-a8369ff9f942)




ResNet50 Model: A pre-trained ResNet50 model fine-tuned for the classification task.

## Model Evaluation
Each model was trained for 10 epochs, and the training and validation accuracies and losses were recorded. The best model was selected based on its performance on the validation set.


![8](https://github.com/michalakatarzyna/Alien-vs-Predator/assets/101282499/8aac5151-096f-4cd2-82c3-f66d12d5b798)


## Gradio Interface
The final selected model is deployed using Gradio to create a user-friendly interface for making predictions on custom images. The interface allows users to upload an image and get predictions on whether it contains a Predator or an Alien.


![9](ce1f6ee80e41a388cf.gradio.live)



Note: The code provided in the notebook is used to implement the above-mentioned tasks. The process of data preprocessing, EDA, model training, hyperparameter tuning, and evaluation is demonstrated using the code in the notebook.
