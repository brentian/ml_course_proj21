import numpy as np
import pandas as pd
#正常资产用C表示，Mn表示逾期N期，用30天来定义一个M：M1逾期一期，M2逾期二期，M3逾期三期，M4逾期四期，M5逾期五期，M6逾期六期，Mn+表示逾期N期(含)以上，M7+表示逾期期数 >=M7。
pos = pd.read_csv('data/POS_CASH_balance.csv')
pos['DPD_DAY'] = (pos['SK_DPD'] - pos['SK_DPD_DEF'])
bin = [-1, 0, 30, 60, 90, 120, 150, 180, 210, 100000]
label = ['C', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M7+']
pos['Default_BINNED'] = pd.cut(pos['DPD_DAY'], bins=bin, labels=label)
i = pos.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'Default_BINNED']]
i = pd.get_dummies(i, columns=['Default_BINNED'])
cat_columns = i.columns.drop(['SK_ID_CURR', 'SK_ID_PREV'])
cat_agg = {}
for a in cat_columns:
    cat_agg[a] = ['mean', 'sum']
i_agg = i.groupby(['SK_ID_CURR', 'SK_ID_PREV'],
                  as_index=False).agg({**cat_agg})
tem = ['SK_ID_CURR', 'SK_ID_PREV']
for i in i_agg.columns.tolist():
    if i[0] != 'SK_ID_CURR' and i[0] != 'SK_ID_PREV':
        tem.append(i[0] + '_' + i[1])
i_agg.columns = pd.Index(tem)
#计算每个贷款人历史贷款逾期状态（C,M1,M2...）的总次数及每种状态的占比
cat_columns = i_agg.columns.drop(['SK_ID_CURR', 'SK_ID_PREV'])
cat_agg = {}
for a in cat_columns:
    cat_agg[a] = ['mean', 'sum']
i_fin = i_agg.groupby(['SK_ID_CURR'], as_index=False).agg({**cat_agg})
tem = ['SK_ID_CURR']
for i in i_fin.columns.tolist():
    if i[0] != 'SK_ID_CURR':
        tem.append(i[0] + '_' + i[1])
i_fin.columns = pd.Index(tem)
#去掉无用列
i_fin.drop([
    'Default_BINNED_C_sum_mean', 'Default_BINNED_C_mean_sum',
    'Default_BINNED_M1_sum_mean', 'Default_BINNED_M1_mean_sum',
    'Default_BINNED_M2_sum_mean', 'Default_BINNED_M2_mean_sum',
    'Default_BINNED_M3_sum_mean', 'Default_BINNED_M3_mean_sum',
    'Default_BINNED_M4_sum_mean', 'Default_BINNED_M4_mean_sum',
    'Default_BINNED_M5_sum_mean', 'Default_BINNED_M5_mean_sum',
    'Default_BINNED_M6_sum_mean', 'Default_BINNED_M6_mean_sum',
    'Default_BINNED_M7_sum_mean', 'Default_BINNED_M7_mean_sum',
    'Default_BINNED_M7+_sum_mean', 'Default_BINNED_M7+_mean_sum'
],
           axis=1,
           inplace=True)
i_fin.to_csv("proc/zwt.ft_POS_CASH.csv", index=None)
