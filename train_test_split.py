import sys
import os
import pandas as pd
from tqdm.autonotebook import tqdm

MAIN_AGG = 'final_ft_application_all.csv'
TRAIN_AGG = 'final_ft_application_train.csv'
TEST_AGG = 'final_ft_application_test.csv'

TARGET_COL_NAME = "TARGET"
MAIN_COL_NAME = "SK_ID_CURR"
BASIC_DTYPES = {"SK_ID_CURR": str, "SK_ID_PREV": str, "SK_ID_BUREAU": str}

if __name__ == '__main__':
    process_directory = sys.argv[1]
    output_directory = sys.argv[2]
    ft_files = os.listdir(process_directory)
    df_main = pd.read_csv(os.path.join(process_directory, MAIN_AGG),
                          dtype=BASIC_DTYPES).set_index(MAIN_COL_NAME)
    idx_test = df_main['TARGET'].isna()

    df_test = df_main[idx_test]
    df_train = df_main[~idx_test]
    df_test.to_csv(f"{output_directory}/{TEST_AGG}")
    df_train.to_csv(f"{output_directory}/{TRAIN_AGG}")