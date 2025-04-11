import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeltaXStopper

def generate_augmented_datasets(train_set: pd.DataFrame,
                                 test_set: pd.DataFrame):
    """
    Generates augmented datasets by combining the original train and test sets with additional fake and true news data.

    Parameters:
    -----------
    train_set (pd.DataFrame): The original training dataset.
    test_set (pd.DataFrame): The original test dataset.

    Returns:
    --------
    tuple: A tuple containing the augmented training dataset and the augmented test dataset.
    """   
    some_train_fake_news = pd.read_csv("../data/some_train_fake_news.csv", index_col=0)
    some_train_true_news = pd.read_csv("../data/some_train_true_news.csv", index_col=0)

    some_test_fake_news = pd.read_csv("../data/some_test_fake_news.csv", index_col=0)
    some_test_true_news = pd.read_csv("../data/some_test_true_news.csv", index_col=0)

    some_train_fake_news = (some_train_fake_news
                            .rename(columns={"text": "drop", "rewritten_text": "text"})
                            .drop(["drop"], axis=1))
    some_train_true_news = (some_train_true_news
                            .rename(columns={"text": "drop", "rewritten_text": "text"})
                            .drop(["drop"], axis=1))
    augmented_train_set = (pd.concat([train_set, some_train_fake_news, some_train_true_news])
                           .reset_index()
                           .drop(["index"], axis=1))
    
    some_test_fake_news = (some_test_fake_news
                            .rename(columns={"text": "drop", "rewritten_text": "text"})
                            .drop(["drop"], axis=1))
    some_test_true_news = (some_test_true_news
                            .rename(columns={"text": "drop", "rewritten_text": "text"})
                            .drop(["drop"], axis=1))
    augmented_test_set = (pd.concat([test_set, some_test_fake_news, some_test_true_news])
                           .reset_index()
                           .drop(["index"], axis=1))

    return augmented_train_set, augmented_test_set

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A custom transformer to clean text data using a specified cleaning function.

    Parameters:
    -----------
    cleaning_function (callable): A function that takes a pandas DataFrame or Series and returns a cleaned version of it.
    """
    def __init__(self, cleaning_function):
        """
        Initializes the TextCleaner with a specified cleaning function.

        Parameters:
        -----------
        cleaning_function (callable): A function that takes a pandas DataFrame or Series and returns a cleaned version of it.
        """
        self.cleaning_function = cleaning_function  

    def fit(self, X, y=None):
        """
        Fits the transformer on the data. This method is a placeholder and does not perform any operations.

        Parameters:
        -----------
        X (pd.DataFrame or pd.Series): The input data to fit the transformer on.
        y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
        --------
        self: The fitted transformer instance.
        """
        return self 

    def transform(self, X):
        """
        Applies the cleaning function to the input data.

        Parameters:
        -----------
        X (pd.DataFrame or pd.Series): The input data to be cleaned.

        Returns:
        --------
        pd.DataFrame or pd.Series: The cleaned data after applying the cleaning function.
        """
        X_ = X.copy()
        return self.cleaning_function(X_)

def train_and_evaluate_classifiers(classifiers,
                                   pipeline,
                                   X_train,
                                   y_train,
                                   cv,
                                   scoring: str="f1"):
    """
    Trains and evaluates multiple classifiers using cross-validation.

    Parameters:
    -----------
    classifiers (dict): A dictionary where keys are classifier names and values are classifier instances.
    pipeline (Pipeline): A scikit-learn pipeline containing preprocessing steps.
    X_train (pd.DataFrame or np.ndarray): The training feature set.
    y_train (pd.Series or np.ndarray): The training target variable.
    cv (int or cross-validation generator): Determines the cross-validation splitting strategy.
    scoring (str, optional): The scoring metric to evaluate the classifiers. Defaults to "f1".

    Returns:
    --------
    dict: A dictionary containing the cross-validation scores for each classifier.
    """
    classifiers_scores = {}
    for k in classifiers.keys():
        pipeline_ = Pipeline(pipeline.steps + [(k, classifiers[k])])
        scores = cross_validate(pipeline_, 
                                X_train, 
                                y_train, 
                                cv=cv, 
                                scoring=scoring, 
                                return_train_score=True, 
                                verbose=0, 
                                n_jobs=-1)
        classifiers_scores[k] = scores
        
    return classifiers_scores


def hypertune_models_bayesianCV(models_to_tune,
                     pipeline,
                     X_train,
                     y_train,
                     cv,
                     scoring: str="f1",
                     random_state: int=42,
                     n_jobs: int=-1,
                     verbose: int=0,
                     n_iter: int=15):
    """
    Performs hyperparameter tuning on multiple models using Bayesian optimization with cross-validation.

    Parameters:
    -----------
    models_to_tune (dict): A dictionary where keys are model names and values are dictionaries containing the model instance and parameter space.
    pipeline (Pipeline): A scikit-learn pipeline containing preprocessing steps.
    X_train (pd.DataFrame or np.ndarray): The training feature set.
    y_train (pd.Series or np.ndarray): The training target variable.
    cv (int or cross-validation generator): Determines the cross-validation splitting strategy.
    scoring (str, optional): The scoring metric to evaluate the models. Defaults to "f1".
    random_state (int, optional): Seed for the random number generator. Defaults to 42.
    n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1 (all processors).
    verbose (int, optional): Controls the verbosity of the output. Defaults to 0.
    n_iter (int, optional): Number of iterations for the Bayesian optimization. Defaults to 15.

    Returns:
    --------
    dict: A dictionary containing the tuned models with their best estimators.
    """
    models_to_tune_ = models_to_tune.copy()
    for k, item in models_to_tune_.items():
        pipeline_ = Pipeline(pipeline.steps + [(k, item["model"])])
    
        bayes_search = BayesSearchCV(pipeline_,
                                     {f"{k}__{kk}": v for kk, v in item["param_space"].items()},
                                     n_iter=n_iter, 
                                     cv=cv,
                                     scoring=scoring,
                                     random_state=random_state,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
    
        bayes_search.fit(X_train, y_train)
    
        models_to_tune_[k].update({"best_model": bayes_search.best_estimator_})

    return models_to_tune_
