from stats_anova import Anova_Bonferroni
import pandas as pd
import numpy as np

df = pd.read_csv('LW.csv')
cols = ['Device','PageNumber','Environment','Mode','FW_Rev','DCC_Rev', 'K_T_Cart', 'K_T_Cart Serial Num','TL_H_K', 'TL_V_K', 'TR_H_K', 'TR_V_K', 'CC_H_K', 'CC_V_K', 'BL_H_K', 'BL_V_K', 'BR_H_K', 'BR_V_K', 'K_H_AVG','K_V_AVG','K_AVG','K_Avg_DTA_PASS','K_H_Avg_DTA_PASS','K_V_Avg_DTA_PASS','DTA_Pass','DTA_Fail_Color']

cols_simp = ['TL_H_K', 'TL_V_K', 'TR_H_K', 'TR_V_K', 'CC_H_K', 'CC_V_K', 'BL_H_K', 'BL_V_K', 'BR_H_K', 'BR_V_K', 'K_H_AVG','K_V_AVG','K_AVG']

cols_1 = ['Device','PageNumber','Environment','Mode','FW_Rev','DCC_Rev', 'K_T_Cart', 'K_T_Cart Serial Num','CC_H_K', 'CC_V_K']

test_df = df[cols_1]
na_1 = test_df[test_df['CC_H_K'].isnull()]
test_df.dropna(subset = ['CC_H_K', 'CC_V_K'], inplace = True)

test_df['phase'] = test_df['Device'].apply(lambda x: 'MT' if 'MT' in x or 'PRT' in x else 'MP')
test_df['product'] = test_df['Device'].apply(lambda x: 'teton' if 'Tet' in x else 'mogami')
test_df['prod&phase'] = test_df['phase'] + test_df['product']

# temp = test_df[['product', 'Device']]
# print (temp.drop_duplicates())
anova_comp = Anova_Bonferroni(test_df['CC_V_K'], test_df['prod&phase'])
anova_comp._anov_basic()
print (anova_comp.anov_cal())
comp = anova_comp.pairwise_cmp(.05)
print (comp)
# print (comp.loc[comp['j_significant'] == True].shape)
