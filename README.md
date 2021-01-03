## Summary of data transformations
1. Decided to aggregate order data on customer level
1. Some variables were aggregated by addition, like cumulative amount paid,
for some variables most popular value for a customer was selected, for some maximum value (like order rank)
1. Aggregated customer dataset was split into train and test sets with 80/20 ratio
1. Numeric variables were MinMax scaled, train set was oversampled with SMOTE algorithm

1. Features were then transformed with Tensorflow functions - categorical 
   features with high cardinality were hashed and bucketized for wide part of 
   deep and wide model, then hashed and embedded for deep part
   
1. All the transformations can be found in `trainer.utils` script, which generates
   
## Summary of the model
1. At first, I experimented with tree-based algorithms (Random Forest and XGBoost).
Given that features are a mix of numeric, ordinal and categorical variables I though
   that wide and deep model might be a good fit
   
1. Tree based models were performing well with ~0.81-0.84 accuracy and ~0.50 
   precision and recall. They were doing worse on positive labels
   
1. Wide and Deep model (DNNLinearCombinedClassifier) is has ~0.76 accuracy with
~0.53 auc after training for just 500 training steps (not even whole epoch). 
   It takes some time due to size of the model
   
1. Results are not impressive, but given the amount of parameters and feature transformations
this model is capable of I believe it could be fine-tuned to surpass tree-based models.
   Sadly I don't have time to do this right now
   
1. I still decided to leave at Wide and Deep model, because I find it much cooler
than `from sklearn import RandomForestClassifier`
   
## How to train the model
1. Open terminal on macOS or linux
1. ```bash
   cd to repo directory
   ```
1. Create new `python=3.8` venv or conda env
1. Run
   ```bash
   pip install -r requirements.txt
   ```
1. Run
   ```bash
   python -m trainer.task
   ```
1. All the data transformations and model training will happen there,
which will take a lot of time.
   
1. My Exploratory Data Analysis can be found in `notebooks` directory. 
   It's a Jupyter Notebook.