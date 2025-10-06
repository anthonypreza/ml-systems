This project aims to build a basic pointwise Learning to Rank (LTR) recommendation system from Goodreads interaction data.

The citation for the dataset used is:

- Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18. [bibtex]
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. [bibtex]

The project contains the full ML lifecycle from Exploratory Data Analysis, Data Preprocessing, Feature Engineering, Data Splitting, Model Development, Training, Validation, Testing, and Deployment via a containerized API.

We initially build a boosted decision tree (XGBoost) as a baseline model, and then move into the development of a Neural Network architecture.

The goal of the model is to predict the probability that a given item (book) is relevant given the user as an input.

For offline scoring, we opt for the mean Average Precision (mAP) as a scoring metric. mAP is recommended when the relevance of an item is binary (relevant or not) in scoring ranking classifiers.

# Book Recommendation from GoodRead Interactions

## Exploratory Data Analysis

## Data Preprocessing

## Feature Engineering

### Running the Feature Engineering Pipeline

## Data Splitting

## Model Development

## Model Evaluation

## Deployment
