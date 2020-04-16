from wl_data import Work_load_Data

data_m = Work_load_Data()
a = data_m.with_time_data()
print (a.head())
print (a.shape)

