Main steps to go through an example project end to end:

1. Look at the big picture.
1. Get the data.
1. Discover and visualize the data to gain insights.
1. Prepare the data for Machine Learning algorithms
1. Select a model and train it
1. Find-tune your model
1. Present your solution
1. Launch, monitor, and maintain your system.

## Working with Real Data

1. Popular open data repositories:
    * UC Irvine Machine Learning Repository
    * Kaggle datasets
    * Amazons's AWS datasets


1. Meta portals(they list open data repositories)
    * http://dataportals.org
    * http://opendatamonitor.eu
    * http://quandl.com

1. Other pages listing many popular open data repositories:
    * Wikipedia's list of Machine Learning datasets
    * Quora.com question
    * Datasets subreddit

## Look at the Big Picture

1. Frame the Problem
    1. Find what exactly is the business objective.
    1. Find what current solution looks like.

1. Select a Performance Measure
    1. A typical performance measure for regression problems is the Root Mean Square Error(RMSE).   
    1. Even though the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to user another function. For example, suppose that there are many outlier districts. In that case, you may consider using the Mean Absolute Error(also called the Average Absolute Deviation;)
        
1. Check the Assumptions

## Get the Data 

https://github.com/ageron/handson-ml

1. Create the Workspace
1. Take a Quick Look at the Data Structure
1. Create a Test Set

### Discover and Visualize the Data to Gain Insights

1. Visualizing Geographical Data
1. Looking for Correlations
1. Experimenting with attribute Combinations

### Prepare the Data for Machine Learning Algorithms
1. Data Cleaning
1. Handling Text and Categorical Attributes
1. Custom Transformers
1. Feature Scaling
    * Min-max scaling

        Min-max scaling (many people call this normalization) is quite simple: values are shifted and rescaled so that the end up ranging from 0 to 1.
    * Standardization

        First it subtracts the mean value (so standardized values always have  a zero mean), and then it divides by the variance so that the resulting distribution has unit variance.
1. Transformation Pipelines

### Select and Train a Model

1. Training and Evaluation on the Training Set
1. Better Evaluation Using Cross-Validation

### Find-Tune Your Model

1. Grid Search
1. Randomized Search
1. Ensemble Methods
1. Analyze the Best Models and Their Errors
1. Evaluate Your System on the Test Set

### Launch, Monitor, and Maintain Your System
