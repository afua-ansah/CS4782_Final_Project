# CS4782_Final_Project

## Introduction
We based our project on the paper 3D Self-Supervised Methods for Medical Imaging by Aiham Taleb et al. It was submitted to the 34th Conference on Neural Information Processing Systems (NeurIPS 2020) and discusses applications of self-supervised computer vision methods to medical imaging tasks.
The paper covers both 2D and 3D methods such as contrastive predictive coding, rotation prediction and jigsaw puzzles, and focuses on downstream tasks in the medical imaging domain such as brain and pancreas tumor segmentation, diabetic retinopathy detection
It compares 2D and 3D methods and assesses gains with respect to efficiency, performance, and convergence speed

Medical imaging is important for disease prevention, diagnosis, and treatment. However, generating expert annotations of 3D medical images at scale is non-trivial, expensive, and time-consuming.
Self-supervised models do not require explicit data labeling, which saves time and money for large unlabeled corpuses or datasets.
Transfer learning from self supervised methods encourage the model to learn semantic representations about the concepts in the data. 
This paper achieves results competitive to state-of-the-art solutions at a fraction of the computational expense.

For our project, we focused on on recreating diabetic retinopathy detection results.
We used the dataset from the Diabetic Retinopathy 2019 Kaggle Challenge, containing roughly 5590 Fundus 2D images, each of which was rated by a clinician on a severity scale of 0 to 4.

## Chosen Result

We chose to implement 3 of the models displayed in this graph: contrastive predictive coding, rotation prediction, and the baseline model. The below figure compares different 2D models used to help detect diabetic retinopathy. The evaluation metric for comparing the models is 5-fold cross-validation accuracy.

<img width="395" alt="Screen Shot 2024-05-15 at 11 18 07 AM" src="https://github.com/afua-ansah/CS4782_Final_Project/assets/34491386/033ef395-5b69-4181-94d9-5a67b9a61323">

## Re-implementation Details

#Baseline

For our baseline model, we trained from scratch without fine-tuning after pre-training on any self-supervised methods.
We used a customized DenseNet121 network for in feature extraction and binary cross entropy loss for the training signal.

#Contrastive Predictive Coding

CPC predicts the representations of image patches below a certain position from those above it, and evaluates the predictions using binary cross entropy loss
It relies on an encoder and context network.
The context network is implemented as an autoregressive network using a Gated Recurent Unit (GRU), and the encoder network is implemented using a customized Densenet121 with a max pooling layer.

#Rotation Prediction

For each image in the dataset, we choose a rotation amount randomly: either 0, 90, 180, or 270 degrees.
Then, we assign a label to that image based on the rotation amount and train a CNN classification model on the rotated images and corresponding labels.
We created a RotationDataset class that randomly assigns a rotation and corresponding one-hot encoding label to each image.
As specified in the paper, we used a DenseNet121 for feature extraction.

#Differences between the paper and our re-implementation

We pretrain on the same Kaggle dataset and perform fine-tuning on labeled subsets of it, while the authors pretrain using 2D Fundus images from UK Biobank (which contains 170K images) and fine tune on the labeled Kaggle dataset.
We pre-train over 10 epochs, while the authors perform pre-training over 1000 epochs.
The authors applied augmentations to the diabetic retinopathy scans prior to pre-training, which we did not include.
In the paper, they use a warm-up procedure during fine-tuning which freezes encoder weights during warm-up epochs, and focuses on training classifiers/decoders.

## Results and Analysis

<img width="597" alt="Screen Shot 2024-05-15 at 11 29 04 AM" src="https://github.com/afua-ansah/CS4782_Final_Project/assets/34491386/77585c66-1d4a-48af-a594-72437621c643">
<img width="395" alt="Screen Shot 2024-05-15 at 11 18 07 AM" src="https://github.com/afua-ansah/CS4782_Final_Project/assets/34491386/033ef395-5b69-4181-94d9-5a67b9a61323">

One of the biggest differences between our results and those of the paper is that our baseline model performed better than CPC and rotation.

## Conclusion and Future Work

Some possible future work we could do on this project includes making changes to make our process more similar to that of the paper, and noting what changes that has. For example, we could freeze encoder weights during warm-up epochs. In addition, we could increase the model capability to cover 3D images as well.

## References

https://arxiv.org/abs/2006.03829
https://arxiv.org/abs/1803.07728
