import sys
import os
import pandas as pd
from tqdm.autonotebook import tqdm
MAIN = 'ft_application_all.csv'
MAIN_AGG = 'final_ft_application_all.csv'

TARGET_COL_NAME = "TARGET"
MAIN_COL_NAME = "SK_ID_CURR"
BASIC_DTYPES = {"SK_ID_CURR": str, "SK_ID_PREV": str, "SK_ID_BUREAU": str}
if __name__ == '__main__':
    process_directory = sys.argv[1]
    output_directory = sys.argv[2]
    ft_files = os.listdir(process_directory)
    df_main = pd.read_csv(os.path.join(process_directory, MAIN),
                          dtype=BASIC_DTYPES).set_index(MAIN_COL_NAME)
    for fp in tqdm(ft_files):
        print(f"{fp} started")
        if fp == MAIN or not fp.endswith(".csv"):
            continue
        fp_full = os.path.join(process_directory, fp)
        df = pd.read_csv(fp_full, dtype=BASIC_DTYPES).set_index(MAIN_COL_NAME)
        df_main = df_main.join(df, how='left')
        print(f"{fp} finished")

    print(f"final shape: {df_main.shape}")
    df_main.to_csv(os.path.join(output_directory, MAIN_AGG))
