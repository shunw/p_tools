from stats_anova import Anova_Bonferroni
import pandas as pd
import numpy as np

df = pd.read_excel('CFT_data_2.xlsx', header=0)
cols = list(df.columns)
remain_cols = ['Media Loading', 'Media Type Setting', 'Tray Side Guide', 'Tray Back Guide', 'Media Fanning', 'sum(# of Page)', 'Jam (Occurances)']
df = df[remain_cols]

m_load = 'Media Loading'
m_type = 'Media Type Setting'
t_side_guid = 'Tray Side Guide'
t_back_guid = 'Tray Back Guide'
m_f = 'Media Fanning'
j_occur = 'Jam (Occurances)'

# print (df[m_type])
# print (df[m_load])

from sklearn.preprocessing import OrdinalEncoder

# df_cat = df[[m_load, m_type, t_side_guid, t_back_guid, m_f]] # can have more than one category col

# ordinal_encoder = OrdinalEncoder()

# df_cat_encoded = ordinal_encoder.fit_transform(df_cat)

# a = ordinal_encoder.categories_ # to check the the array
# print (a)
com = Anova_Bonferroni(df[j_occur], df[m_f])
print (com.anov_cal())