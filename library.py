from __future__ import annotations  # must be first line in your library!

import types
import warnings
from typing import (
    Annotated,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Self,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from annotated_types import Gt
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_halving_search_cv
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import HalvingGridSearchCV, ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

set_config(transform_output="pandas")  # forces built-in transformers to output df


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(
        self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]
    ) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(
            mapping_dict, dict
        ), f"{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead."
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  # column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  # always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."
        assert (
            self.mapping_column in X.columns.to_list()
        ), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  # column legit?
        warnings.filterwarnings(
            "ignore", message=".*downcasting.*"
        )  # squash warning in replace method below

        # now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n"
            )

        # now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(
                f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n"
            )

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[Iterable] = None
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        # self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result


class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one-hot encoding on a specified column.

    This transformer follows the scikit-learn transformer interface and can be
    used in a scikit-learn pipeline. It applies one-hot encoding to a specified
    column, creating new columns for each unique value in the original column.

    Parameters
    ----------
    target_column : str or int
        The name (str) or position (int) of the column to be one-hot encoded.
    dummy_na : bool, default=False
        Whether to create a dummy column for NaN values.
    drop_first : bool, default=False
        Whether to drop the first dummy column to avoid multicollinearity.

    Attributes
    ----------
    target_column : str or int
        The column (by name or position) that will be transformed.
    dummy_na : bool
        Whether to include a dummy column for NaN values.
    drop_first : bool
        Whether to drop the first dummy column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> ohe = CustomOHETransformer('category')
    >>> transformed_df = ohe.fit_transform(df)
    >>> transformed_df
       category_A  category_B  category_C
    0           1           0           0
    1           0           1           0
    2           0           0           1
    3           1           0           0
    """

    def __init__(
        self,
        target_column: Union[str, int],
        dummy_na: bool = False,
        drop_first: bool = False,
    ) -> None:
        """
        Initialize the CustomOHETransformer.

        Parameters
        ----------
        target_column : str or int
            The name (str) or position (int) of the column to be one-hot encoded.
        dummy_na : bool, default=False
            Whether to create a dummy column for NaN values.
        drop_first : bool, default=False
            Whether to drop the first dummy column to avoid multicollinearity.
        """
        self.target_column: Union[str, int] = target_column
        self.dummy_na: bool = dummy_na
        self.drop_first: bool = drop_first

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomOHETransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with one-hot encoding applied to the
            specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if target_column is not in X.
        """
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."
        assert (
            self.target_column in X.columns.to_list()
        ), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'

        X_ = pd.get_dummies(
            X,
            prefix=self.target_column,
            prefix_sep="_",
            columns=[self.target_column],
            dummy_na=self.dummy_na,
            drop_first=self.drop_first,
            dtype=int,
        )
        return X_

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[Iterable] = None
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with one-hot encoding applied to the
            specified column.
        """
        # self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result


class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(
        self, column_list: List[str], action: Literal["drop", "keep"] = "drop"
    ) -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in [
            "keep",
            "drop",
        ], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(
            column_list, list
        ), f"DropColumnsTransformer expected list but saw {type(column_list)}"
        self.column_list: List[str] = column_list
        self.action: Literal["drop", "keep"] = action

    # your code below

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  # always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by either dropping or keeping specified columns.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pandas.DataFrame
            The transformed DataFrame with columns dropped or kept.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame.
        KeyError
            If any column in `column_list` is not found in the input DataFrame.
        """
        assert isinstance(
            X, pd.DataFrame
        ), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead."

        X_ = X.copy()  # Make a copy of the DataFrame to avoid modifying the original

        if self.action == "drop":
            unknown_columns = [col for col in self.column_list if col not in X_.columns]
            if unknown_columns:
                warnings.warn(
                    f"Columns {unknown_columns} not found in DataFrame and will be ignored.",
                    UserWarning,
                )
            X_ = X_.drop(
                columns=[col for col in self.column_list if col in X_.columns],
                errors="ignore",
            )  # errors='ignore' to suppress KeyError
        elif self.action == "keep":
            try:
                X_ = X_[self.column_list]
            except KeyError as e:
                raise KeyError(f"Column {e} not found in the DataFrame.") from e

        return X_

    def fit_transform(self, X: pd.DataFrame, y: None = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        # self.fit(X,y)
        result = self.transform(X)
        return result


class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that removes highly correlated features
    based on Pearson correlation.

    Parameters
    ----------
    threshold : float
        The correlation threshold above which features are considered too highly correlated
        and will be removed.

    Attributes
    ----------
    correlated_columns_ : Optional[List[Hashable]]
        A list of column names (which can be strings, integers, or other hashable types)
        that are identified as highly correlated and will be removed. This attribute
        is set after `fit` is called.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.correlated_columns_: Optional[List[Hashable]] = (
            None  # Initialized during fit
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self:
        """
        Calculates the Pearson correlation matrix and identifies columns to drop.

        Parameters
        ----------
        X : pd.DataFrame
            The input data frame.
        y : Optional[pd.Series], default=None
            Ignored. Present for compatibility.

        Returns
        -------
        Self
            The fitted transformer instance.
        """
        # correlation matrix
        df_corr = X.corr(method="pearson")
        # boolean mask
        masked_df = df_corr.abs() > self.threshold
        # mask lower triangle including diagonal
        upper_mask = np.triu(masked_df, k=1)
        # set correlated columns with any true vals
        self.correlated_columns_ = [
            col for i, col in enumerate(X.columns) if upper_mask[:, i].any()
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the highly correlated columns identified by the `fit` method.

        Parameters
        ----------
        X : pd.DataFrame
            The input data frame.

        Returns
        -------
        pd.DataFrame
            The data frame with correlated columns removed.

        Raises
        ------
        AssertionError
            If `transform` is called before `fit`.
        """
        # Check if fit has been called
        assert (
            self.correlated_columns_ is not None
        ), "CustomPearsonTransformer.transform called before fit."

        # Drop the identified columns
        X_transformed = X.drop(columns=self.correlated_columns_)
        return X_transformed


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """

    def __init__(self, target_column: Hashable):
        self.target_column = target_column
        self.high_wall = None
        self.low_wall = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fits the transformer by calculating the mean and standard deviation of the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : CustomSigma3Transformer
            Fitted transformer.

        Raises
        ------
        AssertionError
            If the input DataFrame is not of type pd.DataFrame or if the target column is not in the DataFrame.
            If the target column is not numeric.
        """
        assert isinstance(
            X, pd.DataFrame
        ), f"expected Dataframe but got {type(X)} instead."
        assert (
            self.target_column in X.columns.to_list()
        ), f"unknown column {self.target_column}"
        assert pd.api.types.is_numeric_dtype(
            X[self.target_column]
        ), f"expected int or float in column {self.target_column}"

        mean = X[self.target_column].mean()
        sigma = X[self.target_column].std()

        self.high_wall = mean + 3 * sigma
        self.low_wall = mean - 3 * sigma

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by clipping the target column to be within the high and low walls.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed DataFrame with the target column clipped.

        Raises
        ------
        AssertionError
            If the transform method is called before fit.
        """
        assert (
            self.high_wall is not None and self.low_wall is not None
        ), "Transformer has not been fitted yet."
        X[self.target_column] = X[self.target_column].clip(
            lower=self.low_wall, upper=self.high_wall
        )
        return X


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """

    def __init__(
        self, target_column: Hashable, fence: Literal["inner", "outer"] = "outer"
    ):
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.outer_low = None
        self.inner_high = None
        self.outer_high = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fits the transformer by calculating the quartiles and IQR of the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : CustomTukeyTransformer
            Fitted transformer.

        Raises
        ------
        AssertionError
            If the input DataFrame is not of type pd.DataFrame or if the target column is not in the DataFrame.
            If the target column is not numeric.
        """
        assert isinstance(
            X, pd.DataFrame
        ), f"expected Dataframe but got {type(X)} instead."
        assert (
            self.target_column in X.columns.to_list()
        ), f"unknown column {self.target_column}"
        assert pd.api.types.is_numeric_dtype(
            X[self.target_column]
        ), f"expected int or float in column {self.target_column}"

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        self.inner_low = q1 - 1.5 * iqr
        self.outer_low = q1 - 3.0 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_high = q3 + 3.0 * iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by clipping the target column based on the specified fence.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X_ : pd.DataFrame
            The transformed DataFrame with the target column clipped according to the chosen fence.

        Raises
        ------
        AssertionError
            If the transform method is called before fit.
            If the fence type specified during initialization is invalid.
        """
        assert (
            self.inner_low is not None
            and self.inner_high is not None
            and self.outer_low is not None
            and self.outer_high is not None
        ), "TukeyTransformer.fit has not been called."

        X_ = X.copy()

        if self.fence == "inner":
            lower_bound = self.inner_low
            upper_bound = self.inner_high
        elif self.fence == "outer":
            lower_bound = self.outer_low
            upper_bound = self.outer_high

        X_[self.target_column] = X_[self.target_column].clip(
            lower=lower_bound, upper=upper_bound
        )
        return X_


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
    """

    def __init__(self, target_column: str):
        """Initializes the CustomRobustTransformer.

        Parameters
        ----------
        target_column : str
            The name of the column to apply the robust scaling to.
        """
        self.target_column = target_column
        self.iqr = None
        self.med = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Self:
        """Compute the median and interquartile range for scaling.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame containing the target column.
        y : pd.Series, optional
            Ignored. Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        Self
            The fitted transformer instance.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        ValueError
            If the target_column is not found in X.
        """

        assert isinstance(X, pd.DataFrame), "Input must be a pandas DataFrame"
        assert (
            self.target_column in X.columns
        ), f"Unrecognized column: {self.target_column}"
        self.iqr = X[self.target_column].quantile(0.75) - X[
            self.target_column
        ].quantile(0.25)
        # avoid division by zero
        if self.iqr == 0:
            self.iqr = 1
        self.med = X[self.target_column].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale the target column using the computed median and IQR.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with the target column scaled.
            If the IQR calculated during fit was 0, returns the original
            DataFrame without scaling the column.

        Raises
        ------
        TypeError
            If the transformer has not been fitted yet (iqr_ is None).
        """
        assert (
            self.iqr is not None
        ), 'This CustomRobustTransformer instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
        X_ = X.copy()
        X_[self.target_column] = (X_[self.target_column] - self.med) / self.iqr
        return X_


PositiveInta = Annotated[int, lambda x: x > 0]


PositiveIntb = Annotated[int, Gt(0)]


class CustomKNNTransformer(BaseEstimator, TransformerMixin):
    """Imputes missing values using K-Nearest Neighbors.

    This transformer wraps the KNNImputer from scikit-learn, ensuring that
    the `add_indicator` parameter is always set to False. It operates on and
    returns pandas DataFrames.

    Parameters
    ----------
    n_neighbors : PositiveIntb, default=5
        Number of neighboring samples to use for imputation. Must be a positive integer.
    weights : Literal["uniform", "distance"], default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': All points in each neighborhood are weighted equally.
        - 'distance': Weight points by the inverse of their distance. Closer
          neighbors of a query point will have a greater influence than
          neighbors which are further away.

    Attributes
    ----------
    n_neighbors : int
        The number of neighbors used for imputation.
    weights : str
        The weight function used ('uniform' or 'distance').
    KNNImputer : KNNImputer
        The underlying scikit-learn KNNImputer instance.
    fitted : bool
        A flag indicating whether the transformer has been fitted.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn import set_config
    >>> set_config(transform_output="pandas") # Ensure pandas output
    >>> data = [[1, 2, np.nan], [3, np.nan, 6], [np.nan, 8, 9], [10, 11, 12]]
    >>> df = pd.DataFrame(data, columns=['a', 'b', 'c'])
    >>> imputer = CustomKNNTransformer(n_neighbors=2)
    >>> imputed_df = imputer.fit_transform(df)
    >>> imputed_df
         a    b     c
    0  1.0  2.0   7.5
    1  3.0  5.0   6.0
    2  6.5  8.0   9.0
    3 10.0 11.0  12.0
    """

    def __init__(
        self,
        n_neighbors: PositiveIntb = 5,
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        """Initialize the CustomKNNTransformer.

        Parameters
        ----------
        n_neighbors : PositiveIntb, default=5
            Number of neighboring samples to use for imputation. Must be a positive integer.
        weights : Literal["uniform", "distance"], default='uniform'
            Weight function used in prediction.

        Raises
        ------
        ValueError
            If n_neighbors is not a positive integer.
        """
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        self.n_neighbors = n_neighbors
        self.weights = weights
        # Instantiate the underlying KNNImputer, hardcoding add_indicator=False
        self.KNNImputer = KNNImputer(
            n_neighbors=n_neighbors, weights=weights, add_indicator=False
        )
        self.fitted = False  # Flag to track if fit has been called

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Self:
        """Fit the imputer on the provided data.

        This method trains the underlying KNNImputer model. It checks if the
        number of neighbors is feasible given the number of samples and stores
        internal state.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer on. Missing values (NaN) are allowed.
        y : Optional[pd.Series], default=None
            Ignored. Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        Self
            The fitted transformer instance.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame.

        Warns
        -----
        UserWarning
            If `n_neighbors` is greater than the number of samples in X.
        """
        assert isinstance(X, pd.DataFrame), "Input must be a pandas dataframe."
        # Warn if n_neighbors is larger than the number of samples
        if self.n_neighbors > len(X):
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is greater than the number of samples ({len(X)}). "
                f"Using {len(X)} neighbors instead.",
                UserWarning,
            )
        # Fit the underlying KNNImputer
        self.KNNImputer.fit(X)
        self.fitted = True  # Mark as fitted
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in X.

        The imputation is done using the model fitted during the `fit` step.

        Parameters
        ----------
        X : pd.DataFrame
            The input data frame with missing values to impute.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed. The output is always a
            pandas DataFrame due to `set_config(transform_output="pandas")`.

        Raises
        ------
        AssertionError
            If `transform` is called before `fit`.
            If the column names or number of columns in X do not match the
            data seen during `fit`.
        """
        # Check if the transformer has been fitted
        assert (
            self.fitted
        ), 'NotFittedError: This CustomKNNTransformer instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'

        # Transform the data using the fitted KNNImputer
        # The output will be a DataFrame because set_config(transform_output="pandas") was called
        X_transformed = self.KNNImputer.transform(X)
        return X_transformed


class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float = 10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead."
        assert (
            self.col in X
        ), f"{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}"
        assert isinstance(
            y, Iterable
        ), f"{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead."
        assert len(X) == len(
            y
        ), f"{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead."

        # Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col + "_target_"
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = (
            X_[self.col].value_counts().to_dict()
        )  # dictionary of unique values in the column col and their counts
        means = (
            X_[target].groupby(X_[self.col]).mean().to_dict()
        )  # dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (
                n + self.smoothing
            )
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(
            X, pd.core.frame.DataFrame
        ), f"{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead."
        assert self.encoding_dict_, f"{self.__class__.__name__}.transform not fitted"

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)


def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200,
) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df,
            labels,
            test_size=0.2,
            shuffle=True,
            random_state=i,
            stratify=labels,  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(
        np.array(Var) - mean_f1_ratio
    ).argmin()  # Index of value closest to mean

    return rs_value, Var


titanic_transformer = Pipeline(
    steps=[
        ("map_gender", CustomMappingTransformer("Gender", {"Male": 0, "Female": 1})),
        (
            "map_class",
            CustomMappingTransformer("Class", {"Crew": 0, "C3": 1, "C2": 2, "C1": 3}),
        ),
        ("target_joined", CustomTargetTransformer(col="Joined", smoothing=10)),
        ("tukey_age", CustomTukeyTransformer(target_column="Age", fence="outer")),
        ("tukey_fare", CustomTukeyTransformer(target_column="Fare", fence="outer")),
        ("scale_age", CustomRobustTransformer(target_column="Age")),
        ("scale_fare", CustomRobustTransformer(target_column="Fare")),
        ("impute", CustomKNNTransformer(n_neighbors=5)),
        (
            "passthrough",
            FunctionTransformer(validate=False),
        ),  # does nothing but does remove warning
    ],
    verbose=True,
)


# Build pipeline and include scalers from last chapter and imputer from this
customer_transformer = Pipeline(
    steps=[
        ("map_os", CustomMappingTransformer("OS", {"Android": 0, "iOS": 1})),
        ("target_isp", CustomTargetTransformer(col="ISP")),
        (
            "map_level",
            CustomMappingTransformer(
                "Experience Level", {"low": 0, "medium": 1, "high": 2}
            ),
        ),
        ("map_gender", CustomMappingTransformer("Gender", {"Male": 0, "Female": 1})),
        ("tukey_age", CustomTukeyTransformer("Age", "inner")),  # from chapter 4
        (
            "tukey_time spent",
            CustomTukeyTransformer("Time Spent", "inner"),
        ),  # from chapter 4
        ("scale_age", CustomRobustTransformer(target_column="Age")),  # from 5
        (
            "scale_time spent",
            CustomRobustTransformer(target_column="Time Spent"),
        ),  # from 5
        ("impute", CustomKNNTransformer(n_neighbors=5)),
        (
            "passthrough",
            FunctionTransformer(validate=False),
        ),  # does nothing but does remove warning
    ],
    verbose=True,
)

titanic_variance_based_split = 107
customer_variance_based_split = 113


def dataset_setup(original_table, label_column_name: str, the_transformer, rs, ts=0.2):
    features = original_table.drop(columns=label_column_name)
    labels = original_table[label_column_name].to_list()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, random_state=rs, test_size=ts, shuffle=True, stratify=labels
    )
    X_train_transformed = the_transformer.fit_transform(X_train, y_train)
    X_test_transformed = the_transformer.transform(X_test)
    x_train_numpy = X_train_transformed.to_numpy()
    x_test_numpy = X_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)
    return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy


def titanic_setup(
    titanic_table,
    transformer=titanic_transformer,
    rs=titanic_variance_based_split,
    ts=0.2,
):
    return dataset_setup(titanic_table, "Survived", transformer, rs, ts)


def customer_setup(
    customer_table,
    transformer=customer_transformer,
    rs=customer_variance_based_split,
    ts=0.2,
):
    return dataset_setup(customer_table, "Rating", transformer, rs, ts)


def threshold_results(thresh_list, actuals, predicted):
    result_df = pd.DataFrame(
        columns=["threshold", "precision", "recall", "f1", "auc", "accuracy"]
    )
    for t in thresh_list:
        yhat = [1 if v >= t else 0 for v in predicted]
        # note: where TP=0, the Precision and Recall both become 0. And I am saying return 0 in that case.
        precision = precision_score(actuals, yhat, zero_division=0)
        recall = recall_score(actuals, yhat, zero_division=0)
        f1 = f1_score(actuals, yhat)
        accuracy = accuracy_score(actuals, yhat)
        auc = roc_auc_score(actuals, predicted)
        result_df.loc[len(result_df)] = {
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "accuracy": accuracy,
        }

    result_df = result_df.round(2)

    # Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
    # Note that fancy_df is not really a dataframe. More like a printable object.
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #800000; color: white; text-align: center",
    }
    properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

    fancy_df = (
        result_df.style.highlight_max(color="pink", axis=0)
        .format(precision=2)
        .set_properties(**properties)
        .set_table_styles([headers])
    )
    return (result_df, fancy_df)


def halving_search(
    model, grid, x_train, y_train, factor=3, min_resources="exhaust", scoring="roc_auc"
):
    halving_cv = HalvingGridSearchCV(
        model,
        grid,  # our model and the parameter combos we want to try
        scoring=scoring,  # from chapter 10
        n_jobs=-1,  # use all available cpus
        min_resources=min_resources,  # "exhaust" sets this to 20, which is non-optimal. Possible bug in algorithm. See https://github.com/scikit-learn/scikit-learn/issues/27422.
        factor=factor,  # double samples and take top half of combos on each iteration
        cv=5,
        random_state=1234,
        refit=True,  # remembers the best combo and gives us back that model already trained and ready for testing
    )
    return halving_cv.fit(x_train, y_train)


def sort_grid(grid):
    sorted_grid = grid.copy()

    # sort values - note that this will expand range for you
    for k, v in sorted_grid.items():
        sorted_grid[k] = sorted(
            sorted_grid[k], key=lambda x: (x is None, x)
        )  # handles cases where None is an alternative value

    # sort keys
    sorted_grid = dict(sorted(sorted_grid.items()))

    return sorted_grid


approvals_transformer = Pipeline(
    steps=[
        # Gender: already categorical 0 or 1
        # Age: numerical, so we might transform to normalize it then apply tukey for outliers
        ("tukey_age", CustomTukeyTransformer(target_column="Age", fence="outer")),
        ("scale_age", CustomRobustTransformer(target_column="Age")),
        # Debt: numerical, so normalize and apply tukey
        ("tukey_debt", CustomTukeyTransformer(target_column="Debt", fence="outer")),
        ("scale_debt", CustomRobustTransformer(target_column="Debt")),
        # YearsEmployed numerical
        (
            "tukey_years_employed",
            CustomTukeyTransformer(target_column="YearsEmployed", fence="outer"),
        ),
        (
            "scale_years_employed",
            CustomRobustTransformer(target_column="YearsEmployed"),
        ),
        # PriorDefault is already categorical 0 or 1
        # Employed is already categorical 0 or 1
        # CreditScore is numerical
        (
            "tukey_credit_score",
            CustomTukeyTransformer(target_column="CreditScore", fence="outer"),
        ),
        ("scale_credit_score", CustomRobustTransformer(target_column="CreditScore")),
        # DriversLicense is already categorical 0 or 1
        # Income is numerical
        ("tukey_income", CustomTukeyTransformer(target_column="Income", fence="outer")),
        ("scale_income", CustomRobustTransformer(target_column="Income")),
        # Impute missing values
        ("impute", CustomKNNTransformer(n_neighbors=5)),
    ],
    verbose=True,
)

approvals_variance_based_split = 174
