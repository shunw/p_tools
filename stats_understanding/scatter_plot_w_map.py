from requests_html import HTMLSession
from lxml import html
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt

def _deal_map():
    '''
    china: 50.00976562500001,-2.6357885741666065,136.23046875000003,57.844750992891
    china_smaller: 
    '''
    min_lon, max_lon = 50.00976562500001, 136.23046875000003
    min_lat, max_lat = -2.6357885741666065, 57.844750992891


def read_each_html_df(fl_name): 
    '''
    read through each html file
    return as df
    '''
    f = open(fl_name, 'r')
    a = ''
    for line in f: 
        if '<td></td>' in line: 
            a += line.replace('<td></td>', '<td>0</td>')
        elif '</td><td>' in line: 
            a += line.replace('</td><td>', '</td>, <td>')
        elif "People's Republic of China" in line: 
            a += line.replace("People's Republic of China", 'China')
        else: 
            a += line
    f.close()
    soup = BeautifulSoup(a, features="lxml")

    for s in soup(['script', 'style']): 
        s.extact()

    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = list(line.strip() for line in text.splitlines())[1:]
    
    # # break multi-headlines into a line each
    # chunks = (phrase.strip() for line in lines for phrase in line.split(","))
    
    # text = refine(text)
    # print ([c for c in lines])
    # print (text[:40])

    # text_ls = text.split('\n')[7:]
    # # print (text_ls[:10])
    province_ls = list()
    city_ls = list()
    
    lat = list()
    lon = list()
    for l in lines: 
        if l:
            l_split = l.split(',')
            

            lat.append(float(l_split[-2].strip()))
            lon.append(float(l_split[-1].strip()))

            if len(l_split) == 6: 
                province_ls.append(l_split[2].strip())
                city_ls.append(l_split[1].strip())
            elif len(l_split) == 7: 
                province_ls.append(l_split[3].strip())
                city_ls.append(l_split[2].strip())
            elif len(l_split) == 5: 
                province_ls.append(l_split[1].strip())
                city_ls.append(l_split[0].strip())
            elif len(l_split) == 4: 
                province_ls.append(l_split[0].strip())
                city_ls.append(l_split[0].strip())
            else: 
                print (l)

    # print (len(lat), len(lon), len(province_ls), len(city_ls))
    df = pd.DataFrame({'city': city_ls, 'prov': province_ls, 'lat': lat, 'lon': lon})
    df2 = pd.DataFrame([['Chongqing', 'Chongqing', 29.4316, 106.912]], columns=list(['city', 'prov', 'lat', 'lon']))
    df = df.append(df2)
    return df


def get_long_leg_people_data():
    df = pd.read_csv('long_leg_people_location_data.csv')
    df.dropna(how = "all", axis = 1, inplace = True)
    df.dropna(how = "all", axis = 0, inplace = True)
    df = df[1:-1]
    df.columns = ['所居城市', '所在人数']
    
    df['所在人数'] = df['所在人数'].apply(lambda x: int(x))
    # print (df.head())
    return df

long_leg_df = get_long_leg_people_data()
# print (long_leg_df.head())
city_pinyin = {'安徽': 'Anhui', '北京': 'Beijing', '福建': 'Fujian', '甘肃': 'Gansu', '广东': 'Guangdong', '广西': 'Guangxi', '贵州': 'Guizhou',  '河北': 'Hebei', '河南':'Henan', '湖北':'Hubei', '湖南':'Hunan', '吉林':'Jilin', '江苏':'Jiangsu', '江西':'Jiangxi', '辽宁':'Liaoning', '内蒙古':'Inner Mongolia', '宁夏':'Ningxia', '山东':'Shandong', '山西':'Shanxi', '陕西':'Shaanxi', '上海':'Shanghai', '四川':'Sichuan', '天津':'Tianjin', '西藏':'Tibet', '新疆': 'Xinjiang', '云南':'Yunnan', '浙江':'Zhejiang', '重庆':'Chongqing'} # '北美洲', '俄罗斯', '海外','美国', '香港', '新加坡',
remove_city_name_ls = ['北美洲', '俄罗斯', '海外','美国', '香港', '新加坡']

long_leg_df = long_leg_df.loc[~long_leg_df['所居城市'].isin(remove_city_name_ls), ]

# refine the long leg df with the latitude and longtitude information
long_leg_df['pin_yin'] = long_leg_df['所居城市'].apply(lambda x: city_pinyin[x])
# print (long_leg_df.head())


city_position_df = read_each_html_df('province_china.html')

# get the median of the lat and lon for some prov
prov_position_median = city_position_df.groupby(['prov']).median().reset_index()
# print (prov_position_median.head())
# # check if there is any mismatch
# for v in city_pinyin.values():
#     if v not in [i for i in city_position_df['prov']]:
#         print (v)
adj = 4.8
min_lon, max_lon = 50.00976562500001, 136.23046875000003
min_lat, max_lat = -2.6357885741666065 + adj, 57.844750992891 + adj

import matplotlib.image as mpimg
china_map = mpimg.imread('china.png')

fig, ax = plt.subplots()

long_leg_df_with_pos = long_leg_df.merge(prov_position_median, left_on= 'pin_yin', right_on = 'prov')


ax.scatter(x = long_leg_df_with_pos['lon'], y = long_leg_df_with_pos['lat'], alpha = .8, s = long_leg_df_with_pos['所在人数'], zorder = 1)
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
BBox = (min_lon, max_lon, min_lat, max_lat)
ax.imshow(china_map, zorder = 0, extent = BBox, aspect = 'equal') #  
# print (long_leg_df_with_pos.describe())
# print (long_leg_df_with_pos.head())
plt.savefig('long_leg_w_map.png', dpi = 300)