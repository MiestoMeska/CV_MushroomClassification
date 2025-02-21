# Mushroom Classification Project

## Introduction

Welcome to the Mushroom Classification Project! Main goal of this project is harnessing the power of transfer learning to tackle the challenge of classifying mushroom types. Utilizing a dataset from Kaggle, our goal is to apply and fine-tune a pre-trained neural network model to accurately identify various classes of mushrooms based on images.

## Project Task

**Main Goal:** Utilize transfer learning to classify mushroom images into their respective categories. This involves selecting an appropriate pre-trained architecture, fine-tuning it on the mushroom dataset, and evaluating its performance.

### Concepts to Explore

In this module, we consolidate our knowledge to address a classification problem. Our journey will include:

- **Transfer Learning:** Applying a pre-trained model to our specific task, leveraging the knowledge it has gained from a different but related problem.

- **Model Tuning:** The focus is on experimenting with various batch sizes, learning rates, and the implementation of gradient accumulation steps. This exploration aims to optimize the model's performance, balance computational resources, and adapt to the dataset's specific challenges.

- **Parameter Exploration:** Through systematic adjustments of batch sizes and learning rates, alongside the strategic use of gradient accumulation, we aim to find the most effective configuration for our model. This process is crucial for enhancing model accuracy and efficiency.

### Project Content

#### Data

1. **Acquisition:** The dataset is available at Kaggle [Mushrooms Classification - Common Genus Images Dataset](https://www.kaggle.com/maysee/mushrooms-classification-common-genuss-images).
   
2. **Exploration:**  [An Exploratory Data Analysis (EDA)](https://github.com/TuringCollegeSubmissions/vruzga-DL.1.5/blob/master/EDA.ipynb) aims to reveal information about the given dataset.

#### Modeling

- **Architecture:** For this project, I have selected **ResNet50** as the model for transfer learning. ResNet50's depth and complexity offer a promising balance between learning capability and computational efficiency, making it an ideal choice for our classification task.

Below is the structure of the ResNet50 model used in this project:

![ResNet50 Structure](https://github.com/TuringCollegeSubmissions/vruzga-DL.1.5/blob/master/assets/ResNet50_structure.JPG)

[Information about each layer of used model.](https://github.com/TuringCollegeSubmissions/vruzga-DL.1.5/blob/master/ResNet50_architecture.ipynb).



Additional information about the [ResNet50 Model.](https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f)

- **Training and Testing:** For an in-depth look at the model training and testing process, including the experimental setup, training logs, and detailed analysis of the model's performance under various configurations, please refer to the [Model Training and Testing Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.1.5/blob/master/Model_training_test.ipynb).

### Conclusions of the Project

Our exploration focused on the effects of varying batch sizes, learning rates, and gradient accumulation steps on a ResNet50 model tasked with mushroom classification. Here are the condensed findings:

#### Without Gradient Accumulation Steps:

- **Learning Rate of 1e-3:** Batch size increases from 16 to 128 improved accuracy from 0.74 to 0.87, suggesting better model generalization with larger batches.
- **Learning Rate of 1e-4:** Outperformed the 1e-3 setting across all batch sizes, achieving peak accuracy of 0.90 at batch size 16.
- **Learning Rate of 1e-5:** High precision and recall were observed, with the best accuracy at a batch size of 32 (0.85), decreasing slightly with larger batches.

#### With Gradient Accumulation Steps:

- **Accumulation Steps = 2 and 3:** Using these steps with a batch size of 64 and a learning rate of 1e-5 markedly improved model performance, enhancing accuracy and maintaining balance across metrics.

These findings highlight the critical role of hyperparameter tuning in optimizing deep learning models. Adjusting batch sizes, learning rates, and using gradient accumulation steps allowed us to refine the model's effectiveness for mushroom classification.

