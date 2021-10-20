import pandas as pd
import numpy as np
import argparse
import sys
from tqdm.autonotebook import tqdm

pd.set_option("display.max_columns", None)

TARGET_COL_NAME = "TARGET"
MAIN_COL_NAME = "SK_ID_CURR"
BASIC_DTYPES = {"SK_ID_CURR": str, "SK_ID_PREV": str, "SK_ID_BUREAU": str}


@np.vectorize
def l1_normalize(val, _max, _min):
    return (val - _min) / (_max - _min + 1e-2)


def general_preprocess(fname, df_main, is_basic=False):
    if is_basic:
        df = df_main.copy().drop(columns=TARGET_COL_NAME)
    else:
        df = df_main.copy()

    print(df.dtypes.sort_values())
    print(f"originally we have {df.columns.shape} columns")
    cols_key_all = ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"]
    cols_key_main = [MAIN_COL_NAME]

    cols_ratio = []
    # categoricals
    cols_cate = [
        k for k, v in df.dtypes.items()
        if v.__str__() == 'object' and k not in cols_key_all
    ]
    # anything else is numerical (regular)
    # numerics
    cols_numeric = [
        k for k in df.columns if k not in cols_key_all and k not in cols_cate
    ]

    df_proc = df.copy().fillna(dict.fromkeys(cols_numeric, 0))

    # one-hot encodings
    cols_one_hot = []

    for col in tqdm(cols_cate):
        _df_col = pd.get_dummies(df[col], prefix=f'ft_{col}')
        for feat_col in _df_col.columns:
            df_proc[feat_col] = _df_col[feat_col]
            cols_one_hot.append(feat_col)

    print(f"categorical encodings finished #{len(cols_cate)} columns")

    feature_num = {
        'numeric': len(cols_numeric),
        'categorical': len(cols_one_hot)
    }
    print(f"{feature_num}")

    # do aggregation
    df_agg = df_proc.groupby(cols_key_main)

    numeric_methods = {"mean", "median"}
    feature_series = {}
    columns_to_agg = cols_numeric + cols_one_hot + cols_ratio
    for col in tqdm(columns_to_agg):
        for method in numeric_methods:
            feature_series[f"{fname}_{col}_{method}"] = df_agg[col].agg(method)
    print(
        f"aggregation / statistics calculation for #{len(columns_to_agg)} columns: @{len(numeric_methods)} statistics."
    )

    df_final = pd.DataFrame(feature_series)
    for col in tqdm(df_final.columns):
        _max, _min = df_final[col].max(), df_final[col].min()
        df_final[col] = l1_normalize(df_final[col], _max, _min)
    if is_basic:
        df_final[TARGET_COL_NAME] = df_main.set_index(
            MAIN_COL_NAME)[TARGET_COL_NAME]
    return df_final


parser = argparse.ArgumentParser("home.credit.preprocess")
parser.add_argument("--fname", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--is_basic", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    fname = args.fname
    data_dir = args.data_dir
    output_dir = args.output_dir
    bool_is_basic = args.is_basic
    data = f'{data_dir}/{fname}.csv'
    if bool_is_basic:
        df_main = pd.read_csv(data, dtype=BASIC_DTYPES)
    else:
        df_main = pd.read_csv(data, dtype=BASIC_DTYPES)

    df_final = general_preprocess(fname, df_main, bool_is_basic)

    print(f"start to save ... ft_{fname}.csv")
    df_final.to_csv(f"{output_dir}/ft_{fname}.csv")
