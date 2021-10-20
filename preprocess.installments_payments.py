import numpy as np
import pandas as pd
import re

# 导入数据
data = pd.read_csv('data/installments_payments.csv')
data_agg = data.groupby(['SK_ID_CURR', 'SK_ID_PREV', "NUM_INSTALMENT_NUMBER"],
                        as_index=False).agg({
                            'DAYS_INSTALMENT': ['mean'],
                            'DAYS_ENTRY_PAYMENT': ['max'],
                            'AMT_INSTALMENT': ['mean'],
                            'AMT_PAYMENT': 'sum'
                        })
tem = []
for i in data_agg.columns.tolist():
    tem.append(i[0])
data_agg.columns = pd.Index(tem)
# 新增一列 DEFAULT_DAY ,表示每期逾期天数
data_agg['DEFAULT_DAY'] = data_agg['DAYS_ENTRY_PAYMENT'] - data_agg[
    'DAYS_INSTALMENT']
# 新增一列 DEFAULT_AMT , 表示每期逾期金额
data_agg['DEFAULT_AMT'] = data_agg['AMT_INSTALMENT'] - data_agg['AMT_PAYMENT']
# 将未到期且尚未还的记录的逾期天数标为0
data_agg['DEFAULT_DAY'][(data_agg['DAYS_INSTALMENT'] > -30)
                        & (data_agg['DAYS_ENTRY_PAYMENT'].isnull())] = 0
# 将正常还款的记录的逾期天数标记成0
data_agg['DEFAULT_DAY'][data_agg['DEFAULT_DAY'].isnull()] = 0
# 逾期天数大于0的记录
tem = data_agg[data_agg['DEFAULT_DAY'] > 0]
# 逾期天数大于0的记录中，计算总天数和平均逾期天数及总金额和平均逾期金额
default_days_agg = tem.groupby('SK_ID_CURR', as_index=False).agg({
    'DEFAULT_DAY': ['sum', 'mean'],
    'DEFAULT_AMT': ['sum', 'mean']
})
# default_days_agg.to_csv("ft_installments_payments", index=None)
df = pd.DataFrame(default_days_agg.to_records())
columns_map = {k: re.sub("[(|)|,|\s|\']", "", k) for k in df.columns}
df.drop(columns=['index']).rename(columns=columns_map).to_csv(
    "proc/zwt.ft_installments_payments.csv", index=None)
