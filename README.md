# Ionosphere Data Classification with Machine Learning

## Project Overview

This project explores the application of Backpropagation Neural Networks (BPNN) to classify ionospheric radar returns from the UCI Machine Learning Repository's Ionosphere dataset. Our goal is to enhance predictive models which are critical for improving the accuracy of global communication and navigation systems.

## Problem Statement

Our task was to identify which hyperparameters, such as activation functions, learning rates, epochs, dropout rates, and optimizers, would yield the highest accuracy in predicting changes in the ionosphere, which can affect radio wave propagation.

## Objectives

1. To develop a machine learning model using the ionosphere dataset.
2. To experiment with different hyperparameters to optimize model accuracy.
3. To achieve the highest prediction accuracy for radar return data from the ionosphere.

## Methodology

- **Data Acquisition**: Sourced from the [UCI Machine Learning Repository's Ionosphere dataset](https://archive.ics.uci.edu/ml/datasets/ionosphere).
- **Data Preprocessing**: Included scaling inputs, encoding output values, and splitting the dataset into training and testing subsets.
- **Model Architecture**: Designed a hybrid neural network combining convolutional and feedforward layers.
- **Hyperparameter Tuning**: Tested various settings to find the optimal configuration for the best model performance.

## Key Findings

- **Best Activation Function**: Leaky ReLU
- **Optimal Learning Rate**: 0.01
- **Effective Epoch Count**: 100
- **Ideal Dropout Rate**: 0.5
- **Top Performing Optimizer**: RMSprop

The final model achieved a high accuracy rate of 98.59%, demonstrating its effectiveness in predicting ionospheric conditions that directly impact satellite communication and navigation.

## Results and Discussion

Our experiments led to the development of a robust model that accurately predicts ionospheric disturbances. Key insights include the importance of dropout in managing overfitting and the impact of activation functions on the model's learning dynamics.

## Conclusion

The project successfully developed a high-accuracy predictive model using BPNN, tailored to the complexities of ionospheric data. Future work could explore the integration of additional data sources and advanced machine learning techniques to further enhance model reliability and performance.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-Learn Library](https://scikit-learn.org/stable/modules/classes.html)
- [UCI Machine Learning Repository - Ionosphere Dataset](https://archive.ics.uci.edu/ml/datasets/ionosphere)

