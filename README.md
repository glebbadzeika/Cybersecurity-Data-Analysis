

# QED-Recruitment-Challenge


This repository contains a solution for cybersecurity data analysis using Gradient Boosting Machine (GBM). The goal of this project was to develop a model capable of predicting cybersecurity alerts. Multiple methodologies were explored, including logistic regression, neural networks with Keras, and decision trees, with GBM ultimately chosen due to its superior performance with this particular dataset.

## Repository Structure

```
.
├── data
│   ├── cybersecurity_training.csv
│   └── cybersecurity_test.csv
└── GBM.py
└── requirements.txt
└──QED_Software.pdf
└── README.md
```

- The `data` directory contains the dataset used for this project, split into training and test data.
- The Python script (`GBM.py`) used to train and evaluate the GBM model.
- `README.md` is this file, which gives an overview of the project and the repository structure.
-‘QED_Software.pdf’ is the PDF-file, that describes the decisions that was made during the solution

## Methodology

1. Logistic regression was initially considered, but due to the complex and non-linear nature of the features in the dataset, this method was ruled out.

2. A neural network model with Keras was implemented. Despite the potential of neural networks, the performance on the dataset was sub-optimal.

3. A decision tree model was used, which was capable of capturing non-linear relationships, but it easily overfitted and performed poorly on unseen data.

4. Given the shortcomings of the previous models, a Gradient Boosting Machine (GBM) was used. GBM is a powerful ensemble method that builds new predictors to correct the residual errors of the prior predictor, reducing both bias and variance. The GBM model provided significantly improved results compared to the other models.

## Running the Code

To run the code,  run the following command:

```sh
python GBM.py
```

## Dependencies

This project uses the following Python libraries:

- pandas
- numpy
- sklearn
- xgboost

Ensure these are installed before running the code.

