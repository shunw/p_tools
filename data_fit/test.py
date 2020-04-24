from wl_data import Training_Testing_data_get

data_m = Training_Testing_data_get()
# a = data_m.df_prepared
# b, c = data_m.proj_each_month()
# # print (a.loc[(a['p_name'] == 'Teton') & (a['phase'] == 'MT'), ])
# # print (b.loc[b['p_name'] == 'Teton', ])
# print (b.head())
# print (c.head())
a, b, c, d = data_m.training_test_split()
print (a.shape, b.shape, c.shape, d.shape)
