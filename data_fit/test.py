from wl_data import Work_load_Data

data_m = Work_load_Data()
a = data_m.df_prepared.copy()
print (a.loc[(a['p_name'] == 'Teton') & (a['phase'] == 'MP'), ])

# print (a.head())

# print (data_m.w_begin_time_tradition())
