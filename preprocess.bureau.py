import pandas as pd
from preprocess import general_preprocess
pd.set_option("display.max_columns", None)


def preprocess_bureau_balance(data_comp):
    df = pd.read_csv(data_comp, dtype={"SK_ID_CURR": str, "SK_ID_BUREAU": str})
    df_final = df.groupby(['SK_ID_BUREAU', 'STATUS'
                           ])['MONTHS_BALANCE'].count().unstack('STATUS',
                                                                fill_value=0)
    rename_columns = {k: f'bureau_ba_type_{k}' for k in df_final.columns}

    df_final = df_final.rename(columns=rename_columns)
    return df_final


if __name__ == "__main__":
    fname = "bureau"
    data = 'data/bureau.csv'
    data_comp = 'data/bureau_balance.csv'
    df_comp = preprocess_bureau_balance(data_comp)
    df_main = pd.read_csv(data, dtype={"SK_ID_CURR": str, "SK_ID_BUREAU": str})
    df = pd.merge(df_main, df_comp, how='left', on='SK_ID_BUREAU')

    df_final = general_preprocess(fname, df)

    print(f"start to save ... ft_{fname}.csv")
    df_final.to_csv(f"proc/ft_{fname}.csv")
