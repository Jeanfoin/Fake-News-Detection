from typing import (
    Callable,
    Iterable,
    Tuple,
    Union,
    Dict,
    List,
    Optional
)
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches
from matplotlib.transforms import blended_transform_factory

def process_protocol(args):
    """
    Processes a protocol by applying a heuristic model scorer to a DataFrame column.

    Parameters:
    args (tuple): A tuple containing protocol_key (str), df (pd.DataFrame), and woe_table (pd.DataFrame)

    Returns:
    tuple: A tuple with protocol_key and the transformed DataFrame column
    """
    protocol_key, df, woe_table = args
    scorer = HeuristicModel._generate_scorer(woe_table)
    return protocol_key, df[protocol_key].apply(scorer)

class HeuristicModel:
    """
    A heuristic model using Weight of Evidence (WOE) and Information Value (IV) for feature scoring
    and predictive analysis.
    """
    def __init__(self):
        self.features = {}
        self.protocols = {}
        self.tables = {}



    def merge(self, other, inplace=False):
        """
        Merge another HeuristicModel into this one (optionally in-place).

        Parameters:
        other (HeuristicModel): The model to merge with the current one.
        inplace (bool): Whether to perform the merge in-place.

        Returns:
        HeuristicModel: The merged model, or self if inplace=True.
        """
        if not isinstance(other, HeuristicModel):
            raise TypeError("Can only merge with another HeuristicModel")

        if inplace:
            self.features.update(other.features)
            self.protocols.update(other.protocols)
            self.tables.update(other.tables)
            return self
        else:
            merged_model = HeuristicModel()
            merged_model.features = {**self.features, **other.features}
            merged_model.protocols = {**self.protocols, **other.protocols}
            merged_model.tables = {**self.tables, **other.tables}
            return merged_model
    

    def _compute_woe_table(self, 
                       df: pd.DataFrame, 
                       feature: str,
                       bins: Union[int, Iterable[float]],
                       eps: float=1e-4):
        """
        Computes the Weight of Evidence (WOE) table for a given feature.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the feature.
        feature (str): The feature column name.
        bins (Union[int, Iterable[float]]): The number of bins or explicit bin edges.
        eps (float): A small constant to avoid division by zero.

        Returns:
        pd.DataFrame: A DataFrame containing the computed WOE values and IV scores.
        """
        df["bins"] = pd.cut(df[feature], bins=bins, right=True) 
        res = df.groupby('bins', observed=True).agg(count=(feature, 'count'), sum=(self.target, 'sum')).reset_index()
        tot_event = res["sum"].sum()
        tot_nonevent = res["count"].sum() - res["sum"].sum()
        res["event%"] = 100 * res["sum"] / tot_event
        res["nonevent%"] = 100 * (res["count"] - res["sum"]) / tot_nonevent
        res["WOE"] = np.log((res["event%"] + eps) / (res["nonevent%"] + eps))
        res["IV"] = res["WOE"] * (res["event%"] - res["nonevent%"])
        return res

    @staticmethod
    def _generate_scorer(woe_table: pd.DataFrame):
        """
        Creates a scoring function based on bins and their WOE values.
        
        Parameters:
        woe_table (pd.DataFrame): A DataFrame containing bin ranges and corresponding WOE values.
        
        Returns:
        function: A function that takes a value and returns the corresponding WOE score.
        """
        def scorer(x):
            score = 0
            flag = 0
            for b, woe in zip(woe_table["bins"], woe_table["WOE"]):
                if b.left < x <= b.right:
                    score += woe
                    flag = 1
                    break

            if flag == 0:
                if x < woe_table["bins"].iloc[0].left:
                    score = woe_table["WOE"].iloc[0]
                elif x > woe_table["bins"].iloc[-1].right:
                    score = woe_table["WOE"].iloc[-1]
            
            return score
        
        return scorer
    
    def setup_feature_protocol(self,
                               df: pd.DataFrame,
                               feature: str,
                               bins: Union[int, Iterable[float]]):
        """
        Sets up the feature protocol by computing the Weight of Evidence (WOE) table,
        generating a scorer function, and storing them in class attributes.

        Parameters:
        df (pd.DataFrame): The input dataframe containing the feature
        feature (str): The name of the feature to process
        bins (Union[int, Iterable[float]]): The binning strategy for the feature
        """
        woe_table = self._compute_woe_table(df, feature, bins)
        self.tables[feature] = woe_table

        scorer = self._generate_scorer(woe_table)
        self.protocols[feature] = scorer

        self.features[feature] = bins

    
    def setup_model(self, 
                    df: pd.DataFrame,
                    target: str,
                    features: Optional[Union[List[str], Dict[str, Union[int, Iterable[float]]]]]=None,
                    bins: Optional[int]=10):
        """
        Sets up the model by defining the target variable and setting up feature protocols.

        Parameters:
        df (pd.DataFrame): The input dataframe
        target (str): The target variable name
        features (Optional[Union[List[str], Dict[str, Union[int, Iterable[float]]]]], optional):
            The features to be used for modeling. If a dictionary is provided,
            it specifies binning strategies. Defaults to None, meaning all numerical
            features except the target are used.
        bins (Optional[int], optional): The default number of bins for discretization. Defaults to 10.
        """
        df_ = df.copy()    

        self.target = target

        if isinstance(features, dict):
            for f, b in features.items():
                self.setup_feature_protocol(df_, f, b)
        elif isinstance(features, list):
            for f in features:
                self.setup_feature_protocol(df_, f, bins)
        elif features is None:
            features = [f for f in df.select_dtypes(include=np.number).columns.tolist() 
                        if f != self.target]
            for f in features:
                self.setup_feature_protocol(df_, f, bins)

    def get_IV_scores(self, 
                      sort: bool=True):
        """
        Computes Information Value (IV) scores for all features.

        Parameters:
        sort (bool, optional): If True, sorts the features by IV score in descending order. Defaults to True.

        Returns:
        dict: A dictionary with feature names as keys and their IV scores as values.
        """
        IV_scores = {}
        for k, t in self.tables.items():
            IV_scores[k] = t["IV"].sum(axis=0)
        if sort:
            return {k: v for k, v in sorted(IV_scores.items(), key=lambda item: item[1], reverse=True)}
        else:
            return IV_scores

    def predict_score(self, df: pd.DataFrame, features: Iterable[str]=None, n_jobs: int=1):
        """
        Predicts scores for a given dataframe based on the defined feature protocols.

        Parameters:
        df (pd.DataFrame): The input dataframe
        features (Iterable[str], optional): The features to be used for prediction. Defaults to None.
        n_jobs (int, optional): The number of parallel processes to use. Defaults to 1.

        Returns:
        pd.DataFrame: A dataframe containing the calculated scores for each feature.
        """
        scores = {}
        scores_df = pd.DataFrame(index=df.index)

        if features is None:
            features = self.tables.keys()

        common_features = list(set(features) & set(df.columns))
        print(f"Features used for predictions are \n{common_features}.")
        excluded_df_features = list(set(df.columns).difference(features))
        print(f"The following features were in the dataset but not in the model: \n{excluded_df_features}")
        excluded_model_features = list(set(features).difference(df.columns))
        print(f"The following features were in the model but not the dataset: \n{excluded_model_features}")

        n_processors = cpu_count()
        if n_jobs <= 0 or n_jobs > n_processors:
            n_jobs = max(1, n_processors - 1)

        # Prepare arguments for parallel processing
        args = [(protocol_key, df, self.tables[protocol_key]) for protocol_key in common_features]

        with Pool(processes=n_jobs) as pool:
            results = pool.map(process_protocol, args)

        for protocol_key, result_series in results:
            scores[protocol_key] = result_series

        scores_df = pd.concat([scores_df, pd.DataFrame(scores)], axis=1)

        return scores_df
    
    def predict(self, df: pd.DataFrame, features: Iterable[str]=None, n_jobs: int=1):
        """
        Makes final binary predictions based on computed scores.

        Parameters:
        df (pd.DataFrame): The input dataframe
        features (Iterable[str], optional): The features to be used for prediction. Defaults to None.
        n_jobs (int, optional): The number of parallel jobs to use. Defaults to 1.

        Returns:
        pd.Series: A series containing binary predictions (0 or 1).
        """
        scores = self.predict_score(df, features, n_jobs)
        return (scores.sum(axis=1) > 0).astype(int)

    def create_protocol_axis(self, 
                             feature: str, 
                             fig=None, 
                             ax=None):
        """Create and configure an axis for a single protocol feature."""
        if feature not in self.features.keys():
            print(f"{feature} is not part of the protocol... Skipped!")
            return None, None
        
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        xmin = self.tables[feature].iloc[0]["bins"].left
        xmax = self.tables[feature].iloc[-1]["bins"].right
        ymin = self.tables[feature]["WOE"].min()
        ymax = self.tables[feature]["WOE"].max()
        rmax = max(abs(ymax), abs(ymin))
        border = 0.1 * rmax
        IV_max = self.tables[feature]["IV"].max()
        
        ax2 = ax.twinx()
        ax.set_zorder(1)
        ax2.set_zorder(1)
        
        x = np.linspace(xmin, xmax, 1000)
        scores = [self.protocols[feature](val) for val in x]
        xticks = set()
        
        for idx, row in self.tables[feature].iterrows():
            xticks.add(row["bins"].left)
            xticks.add(row["bins"].right)
            ax.axvline(x=row["bins"].left, color='red', linestyle='--', alpha=0.3)
            IV_patch1 = patches.Rectangle((row["bins"].left, row["event%"]), row["bins"].length, 
                                         100 - row["event%"], color="g", alpha=row["IV"] / IV_max / 2, zorder=-1)
            IV_patch2 = patches.Rectangle((row["bins"].left, -100), row["bins"].length, 
                                         100 - row["nonevent%"], color="g", alpha=row["IV"] / IV_max / 2, zorder=-1)
            revent = patches.Rectangle((row["bins"].left, 0), row["bins"].length, 
                                      row["event%"], color="r", alpha=0.5, zorder=-1, ec="w")
            rnonevent = patches.Rectangle((row["bins"].left, 0), row["bins"].length, 
                                         -row["nonevent%"], color="b", alpha=0.5, zorder=-1, ec="w")

            woe_arrow = FancyArrowPatch((row["bins"].mid, 0), (row["bins"].mid, row["WOE"]), arrowstyle='-', linewidth=2, color='black')
            ax.add_patch(woe_arrow)
            ax.plot(row["bins"].mid, row["WOE"], 'o', markersize=5, color="k")
            
            ax2.add_patch(IV_patch1)
            ax2.add_patch(IV_patch2)
            ax2.add_patch(revent)   
            ax2.add_patch(rnonevent)


        trans = blended_transform_factory(ax2.transAxes, ax2.transData)
        y_range = ax2.get_ylim()
        mid_y = (y_range[0] + y_range[1]) / 2
        ax2.text(1.085, 100, "1", ha='center', va='center', fontsize=20,
                 transform=trans)
        ax2.text(1.085, -100, "0", ha='center', va='center', fontsize=20,
                 transform=trans)
        arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5, shrinkA=3, shrinkB=3)
        ax2.annotate("", xy=(1.085, 90), xytext=(1.085, 30),
                    xycoords=trans, textcoords=trans,
                    arrowprops=arrow_props, annotation_clip=False)
        ax2.annotate("", xy=(1.085, -90), xytext=(1.085, -30),
                    xycoords=trans, textcoords=trans,
                    arrowprops=arrow_props, annotation_clip=False)
        ax.axhline(y=0, color='green', linestyle='--', alpha=1, zorder=3)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((-rmax - border, rmax + border))
        ax2.set_ylim((-100, 100))  
        ax2.set_ylabel("Class\ndistribution", fontsize=16)
        ax.set_xlabel(f"{feature}", fontsize=16)
        ax.set_ylabel("WOE score", fontsize=16)
        ax.set_xticks(list(xticks))
        ax2.set_yticks([-100, -50, 0, 50, 100])
        ax2.set_yticklabels(["100%", "50%", "0%", "50%", "100%"])
        
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax2.tick_params(axis="y", which="major", labelsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
        ax.set_title(f"{feature} heuristic protocol", fontsize=22)

        return fig, ax, ax2

    def plot_protocol(self, 
                      features: Union[str, Iterable[str]], 
                      figsize: Tuple=(12, 4)):
        """Plot protocols for multiple features using the builder approach."""        
        if isinstance(features, list):
            if len(features) == 1:
                fig, ax, _ = self.create_protocol_axis(features[0])
                fig.tight_layout()
                return fig, ax
            else:
                fig, axs = plt.subplots(len(features), 1, figsize=(12, 4 * len(features)))
                
                for i, feature in enumerate(features):
                    if feature not in self.features.keys():
                        continue
                        
                    _, _, _ = self.create_protocol_axis(feature, fig=fig, ax=axs[i])
                
                fig.tight_layout()
                return fig, axs
        elif isinstance(features, str):
                fig, ax, _ = self.create_protocol_axis(features)
                fig.tight_layout()
                return fig, ax
        else:
            print("Wrong argument for features!")
            return None, None
