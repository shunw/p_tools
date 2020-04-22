from wl_data import Work_load_Data

data_m = Work_load_Data()
a = data_m.df_prepared
b = data_m.proj_each_month()
# print (a.loc[(a['p_name'] == 'Teton') & (a['phase'] == 'MT'), ])
# print (b.loc[b['p_name'] == 'Teton', ])
