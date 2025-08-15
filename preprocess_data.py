"""
Bank Marketing Preprocessing Pipeline
=====================================

A compact, reproducible preprocessing pipeline for the term-deposit subscription
prediction task (Bank Marketing dataset). It includes:

1) Train/val/test split
2) Leak-safe basic feature engineering
3) OneHot-encoding of categorical features + scaling of numeric features
4) Feature selection:
   - Cluster top-1 (based on correlation clusters) with LightGBM feature importance
   - Additional cleanup of highly correlated pairs (|corr| > 0.95)
5) Optional train balancing via SMOTE/SMOTE-Tomek
6) Consistent transformation of new data (production)

Notes
-----
- Binning thresholds for `cons.conf.idx` / `cons.price.idx` are learned on TRAIN
  only and then applied to VAL/TEST/NEW to avoid leakage.
- Interaction features are added conservatively to avoid dimensionality explosion.
"""

from __future__ import annotations
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# --- additions for Cluster top-1 & cleanup ---
from lightgbm import LGBMClassifier
from scipy.cluster import hierarchy
from sklearn.metrics import pairwise_distances

# ----------------------------- Public API ----------------------------- #
__all__ = [
    "split_train_val_test",
    "separate_inputs_targets",
    "fit_scaler",
    "scale_numeric_features",
    "fit_encoder",
    "encode_categorical_features",
    "preprocess_inputs",
    "preprocess_data",
    "preprocess_new_data",
    "get_selected_features",
]

# ----------------------- Globals / Runtime State ---------------------- #
_LAST_SELECTED_FEATURES: Optional[List[str]] = None  # final list of selected features
_BIN_PARAMS: dict = {}  # binning thresholds for cons.conf.idx and cons.price.idx


# ================================ Utils ================================ #

def get_selected_features() -> Optional[List[str]]:
    """
    Return the final list of features selected by the latest pipeline run
    (after Cluster top-1 and post-cluster high-correlation cleanup),
    or None if selection has not been performed yet.

    Returns
    -------
    Optional[List[str]]
        Feature names or None.
    """
    return _LAST_SELECTED_FEATURES


def split_train_val_test(
    df: pd.DataFrame,
    target_col: str,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train/validation/test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with target.
    target_col : str
        Target column name.
    val_size : float, default=0.2
        Validation fraction (from the residual after taking out test).
    test_size : float, default=0.2
        Test fraction from the full dataset.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    (train_df, val_df, test_df) : tuple of pd.DataFrame
    """
    train_temp, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_temp,
        test_size=val_size / (1 - test_size),
        stratify=train_temp[target_col],
        random_state=random_state
    )
    return train_df, val_df, test_df


def separate_inputs_targets(
    df: pd.DataFrame, input_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with features and target.
    input_cols : List[str]
        Feature columns.
    target_col : str
        Target column name.

    Returns
    -------
    (X, y) : Tuple[pd.DataFrame, pd.Series]
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: List[str]) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on numeric columns (TRAIN only).

    Parameters
    ----------
    train_inputs : pd.DataFrame
        Train features.
    numeric_cols : List[str]
        Numeric feature names.

    Returns
    -------
    MinMaxScaler
    """
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    return scaler


def scale_numeric_features(scaler: MinMaxScaler, inputs: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Scale numeric columns with a fitted scaler (in-place).

    Parameters
    ----------
    scaler : MinMaxScaler
        Fitted scaler.
    inputs : pd.DataFrame
        Data to scale.
    numeric_cols : List[str]
        Numeric column names.
    """
    if len(numeric_cols) > 0:
        inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])


def fit_encoder(train_inputs: pd.DataFrame, categorical_cols: List[str]) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on categorical columns (TRAIN only).

    Parameters
    ----------
    train_inputs : pd.DataFrame
        Train features.
    categorical_cols : List[str]
        Categorical column names.

    Returns
    -------
    OneHotEncoder
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
        .fit(train_inputs[categorical_cols])
    return encoder


def encode_categorical_features(
    encoder: OneHotEncoder, inputs: pd.DataFrame, categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Encode categorical columns using a fitted OneHotEncoder.

    Parameters
    ----------
    encoder : OneHotEncoder
        Fitted encoder.
    inputs : pd.DataFrame
        Input features.
    categorical_cols : List[str]
        Categorical column names.

    Returns
    -------
    pd.DataFrame
        Input features with one-hot encoded categories.
    """
    if len(categorical_cols) == 0:
        return inputs
    encoded = encoder.transform(inputs[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=inputs.index
    )
    inputs = inputs.drop(columns=categorical_cols)
    return pd.concat([inputs, encoded_df], axis=1)


def preprocess_inputs(
    df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[MinMaxScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = False
) -> Tuple[pd.DataFrame, MinMaxScaler, OneHotEncoder]:
    """
    Full feature preprocessing: numeric scaling + categorical one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    input_cols : List[str]
        All feature columns (after feature engineering).
    numeric_cols : List[str]
        Numeric columns.
    categorical_cols : List[str]
        Categorical columns.
    scaler : Optional[MinMaxScaler], default=None
        Pre-fitted scaler or None.
    encoder : Optional[OneHotEncoder], default=None
        Pre-fitted encoder or None.
    fit : bool, default=False
        If True, fit scaler/encoder; otherwise only transform.

    Returns
    -------
    (X, scaler, encoder) : Tuple[pd.DataFrame, MinMaxScaler, OneHotEncoder]
    """
    inputs = df[input_cols].copy()

    if fit and len(numeric_cols) > 0:
        scaler = fit_scaler(inputs, numeric_cols)
    if len(numeric_cols) > 0:
        scale_numeric_features(scaler, inputs, numeric_cols)

    if fit and len(categorical_cols) > 0:
        encoder = fit_encoder(inputs, categorical_cols)
    if len(categorical_cols) > 0:
        inputs = encode_categorical_features(encoder, inputs, categorical_cols)

    return inputs, scaler, encoder


# ========================= Feature Engineering ========================= #

def _fit_bin_edges(train_df: pd.DataFrame) -> None:
    """
    Compute bin edges (tertiles) for `cons.conf.idx` and `cons.price.idx` on TRAIN
    and store them in the global `_BIN_PARAMS` for reuse on VAL/TEST/NEW.

    Parameters
    ----------
    train_df : pd.DataFrame
        TRAIN frame after initial engineering (raw columns).
    """
    global _BIN_PARAMS
    _BIN_PARAMS = {}
    if 'cons.conf.idx' in train_df.columns:
        _BIN_PARAMS['cci_q'] = np.quantile(train_df['cons.conf.idx'].dropna(), [0.33, 0.66]).tolist()
    if 'cons.price.idx' in train_df.columns:
        _BIN_PARAMS['cpi_q'] = np.quantile(train_df['cons.price.idx'].dropna(), [0.33, 0.66]).tolist()


def _map_month_to_num(s: pd.Series) -> pd.Series:
    """
    Map month abbreviations (str) to numbers 1..12; unknowns -> NaN.
    """
    m = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
         'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    return s.astype(str).str.lower().map(m)


def _map_day_to_num(s: pd.Series) -> pd.Series:
    """
    Map weekdays (str) to numbers (mon..fri -> 1..5); unknowns -> NaN.
    """
    d = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}
    return s.astype(str).str.lower().map(d)


def _engineer_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Create additional features. For validation/test/new data, use binning thresholds
    learned on TRAIN and kept in `_BIN_PARAMS`.

    Executed BEFORE one-hot/scaling (on raw columns).

    New features created:
    - contact_cellular (from 'contact')
    - cci_bin (tertiles of cons.conf.idx), cpi_bin (tertiles of cons.price.idx)
    - campaign_log1p, campaign_inv
    - euribor3m_over_cpi
    - contact_cellular_recent = (contact_cellular==1 & pdays<10)
    - Interactions: campaign_x_contact, euribor_x_cci, pdays_x_campaign,
                    contact_x_day

    Parameters
    ----------
    df : pd.DataFrame
        Input frame (after removing 'duration').
    is_train : bool
        If True, (re)fit bin edges on this TRAIN slice; otherwise reuse saved edges.

    Returns
    -------
    pd.DataFrame
        Frame with engineered features.
    """
    out = df.copy()

    #  is_contact_cellular as binary flag (before OHE)
    if 'contact' in out.columns:
        out['is_contact_cellular'] = (out['contact'].astype(str).str.lower() == 'cellular').astype(int)

    #  tertiles for cons.conf.idx / cons.price.idx
    if is_train:
        _fit_bin_edges(out)

    cci_q = _BIN_PARAMS.get('cci_q', None)
    cpi_q = _BIN_PARAMS.get('cpi_q', None)

    if 'cons.conf.idx' in out.columns and cci_q is not None:
        out['cci_bin'] = pd.cut(out['cons.conf.idx'], bins=[-np.inf, cci_q[0], cci_q[1], np.inf],
                                labels=['low','med','high'])
    if 'cons.price.idx' in out.columns and cpi_q is not None:
        out['cpi_bin'] = pd.cut(out['cons.price.idx'], bins=[-np.inf, cpi_q[0], cpi_q[1], np.inf],
                                labels=['low','med','high'])

    #  scale-stabilizing transforms
    if 'campaign' in out.columns:
        out['campaign_log1p'] = np.log1p(out['campaign'])
        out['campaign_inv']   = 1.0 / (out['campaign'] + 1.0)

    if {'euribor3m','cons.price.idx'}.issubset(out.columns):
        out['euribor3m_over_cpi'] = out['euribor3m'] / (out['cons.price.idx'] + 1e-6)

    if {'is_contact_cellular','pdays'}.issubset(out.columns):
        out['is_contact_cellular_recent'] = ((out['is_contact_cellular']==1) & (out['pdays']<10)).astype(int)

    #  interactions (kept lean, age-related removed)
    if {'campaign','is_contact_cellular'}.issubset(out.columns):
        out['campaign_x_contact'] = out['campaign'] * out['is_contact_cellular']

    if {'euribor3m','cons.conf.idx'}.issubset(out.columns):
        out['euribor_x_cci'] = out['euribor3m'] * out['cons.conf.idx']

    if {'pdays','campaign'}.issubset(out.columns):
        out['pdays_x_campaign'] = out['pdays'] * out['campaign']

    if {'is_contact_cellular','day_of_week'}.issubset(out.columns):
        day_num = _map_day_to_num(out['day_of_week']).fillna(0)
        out['contact_x_day'] = out['is_contact_cellular'] * day_num

    return out


# ====================== Correlation-Aware Feature Pick ====================== #

def _select_top_features_by_cluster_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.3,
    random_state: int = 42
) -> List[str]:
    """
    Select exactly one feature (the most important by LightGBM) per
    correlation cluster (average linkage on distance = 1 − |corr|).

    Parameters
    ----------
    X : pd.DataFrame
        Encoded numeric TRAIN features.
    y : pd.Series
        Binary target (0/1).
    threshold : float, default=0.3
        Dendrogram cut distance (smaller -> more clusters).
    random_state : int, default=42
        Random seed for LightGBM.

    Returns
    -------
    List[str]
        Selected feature names.
    """
    #  quick LightGBM for importances
    model = LGBMClassifier(random_state=random_state)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)

    #  absolute correlation matrix
    corr = X.corr().abs().fillna(0.0)

    #  build clusters over pairwise distances of the corr matrix
    link = hierarchy.linkage(pairwise_distances(corr), method='average')
    clusters = hierarchy.fcluster(link, t=threshold, criterion='distance')

    #  pick top-importance feature per cluster
    selected: List[str] = []
    for cid in np.unique(clusters):
        members = [col for col, c in zip(corr.columns, clusters) if c == cid]
        top = importances[members].idxmax()
        selected.append(top)

    return sorted(set(selected))


def _remove_high_corr_after_cluster(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    threshold: float = 0.95,
    random_state: int = 42
) -> List[str]:
    """
    After Cluster top-1, drop the less important (by LightGBM) feature from each
    remaining pair with |corr| > threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Encoded features.
    y : pd.Series
        Binary target.
    features : List[str]
        Current selected features.
    threshold : float, default=0.95
        Correlation threshold for pruning.
    random_state : int, default=42
        Seed for LightGBM.

    Returns
    -------
    List[str]
        Pruned feature list.
    """
    if len(features) < 2:
        return features

    model = LGBMClassifier(random_state=random_state)
    model.fit(X[features], y)
    importances = pd.Series(model.feature_importances_, index=features)

    corr = X[features].corr().abs().fillna(0.0)
    to_remove: set[str] = set()

    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            cval = corr.iloc[i, j]
            if cval > threshold:
                f1, f2 = cols[i], cols[j]
                # remove the less important one
                if importances[f1] < importances[f2]:
                    to_remove.add(f1)
                else:
                    to_remove.add(f2)

    pruned = [f for f in features if f not in to_remove]
    return pruned


# =============================== Pipeline =============================== #

def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = True
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    List[str], MinMaxScaler, OneHotEncoder
]:
    """
    Complete preprocessing pipeline for raw data.

    Steps
    -----
    1) Leak-safe basic engineering (contacted_before), drop 'duration'
    2) Stratified train/val/test split
    3) Feature engineering (binning, interactions, stabilizations) on raw columns
    4) Define `input_cols` including engineered features
    5) OneHot + (optional) scaling
    6) Feature selection: Cluster top-1 + high-corr cleanup
    7) Save final selected features (accessible via `get_selected_features()`)

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original raw data (includes 'y' and raw columns).
    scale_numeric : bool, default=True
        Kept for interface compatibility (MinMaxScaler is used regardless).

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, input_cols, scaler, encoder)
        Processed splits, feature column list (after engineering), and fitted scaler/encoder.
    """
    global _LAST_SELECTED_FEATURES

    #  leak-safe basic features
    raw_df = raw_df.copy()
    # do NOT use 'duration'; create contacted_before upfront
    raw_df['contacted_before'] = np.where(raw_df['pdays'] != 999, 1, 0)
    df = raw_df.drop(columns=[c for c in ['duration'] if c in raw_df.columns])

    #  split
    train_df, val_df, test_df = split_train_val_test(df, target_col='y')

    #  feature engineering on raw columns
    train_df = _engineer_features(train_df, is_train=True)
    val_df   = _engineer_features(val_df,   is_train=False)
    test_df  = _engineer_features(test_df,  is_train=False)

    target_col = 'y'
    input_cols = [c for c in train_df.columns if c != target_col]

    #  separate X/y
    X_train, y_train = separate_inputs_targets(train_df, input_cols, target_col)
    X_val,   y_val   = separate_inputs_targets(val_df,   input_cols, target_col)
    X_test,  y_test  = separate_inputs_targets(test_df,  input_cols, target_col)

    #  map target
    y_train = y_train.map({'yes': 1, 'no': 0})
    y_val   = y_val.map({'yes': 1, 'no': 0})
    y_test  = y_test.map({'yes': 1, 'no': 0})

    #  types (recomputed after engineering)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    #  encode & scale
    X_train, scaler, encoder = preprocess_inputs(
        X_train, input_cols, numeric_cols, categorical_cols, fit=True
    )
    X_val, _, _ = preprocess_inputs(
        X_val, input_cols, numeric_cols, categorical_cols, scaler, encoder
    )
    X_test, _, _ = preprocess_inputs(
        X_test, input_cols, numeric_cols, categorical_cols, scaler, encoder
    )

    #  feature selection: cluster top-1
    selected = _select_top_features_by_cluster_lgbm(
        X_train, y_train, threshold=0.3, random_state=42
    )
    #  post-cluster high-correlation cleanup (|corr| > 0.95)
    selected = _remove_high_corr_after_cluster(
        X_train, y_train, features=selected, threshold=0.95, random_state=42
    )

    _LAST_SELECTED_FEATURES = selected
    X_train = X_train[selected]
    X_val   = X_val[selected]
    X_test  = X_test[selected]

    return X_train, y_train, X_val, y_val, X_test, y_test, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    scale_numeric: bool = True
) -> pd.DataFrame:
    """
    Apply the SAME preprocessing steps to new data that were learned on TRAIN.

    Parameters
    ----------
    new_df : pd.DataFrame
        New raw data, structured like TRAIN (raw columns).
    input_cols : List[str]
        The feature columns returned by `preprocess_data` (includes engineered features).
    scaler : MinMaxScaler
        Scaler fitted on TRAIN.
    encoder : OneHotEncoder
        Encoder fitted on TRAIN.
    scale_numeric : bool, default=True
        Whether to scale numeric columns.

    Returns
    -------
    pd.DataFrame
        Processed features aligned to the training scheme.

    Notes
    -----
    - Uses the saved bin edges in `_BIN_PARAMS`.
    - Before inference, you may want to reindex to the FINAL selected list from
      `get_selected_features()` to ensure exact column matching with the trained model.
    """
    df = new_df.copy()
    df['contacted_before'] = np.where(df.get('pdays', pd.Series(index=df.index)) != 999, 1, 0)

    # Drop 'duration' (same as in TRAIN)
    df = df.drop(columns=[c for c in ['duration'] if c in df.columns])

    # Same feature engineering (bin edges are already stored from TRAIN)
    df = _engineer_features(df, is_train=False)

    # Take exactly the columns expected at the encoding/scaling stage
    df = df.reindex(columns=input_cols, fill_value=0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if scale_numeric and len(numeric_cols) > 0:
        scale_numeric_features(scaler, df, numeric_cols)
    if len(categorical_cols) > 0:
        df = encode_categorical_features(encoder, df, categorical_cols)

    # Optionally narrow to final selected features (recommended right before inference):
    sel = get_selected_features()
    if sel is not None:
        df = df.reindex(columns=sel, fill_value=0)

    return df