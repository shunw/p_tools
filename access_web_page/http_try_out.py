###############
# @author sw
# @date 2018-09-28
#######################

import socket
import urllib.request, urllib.parse, urllib.error
import re
from bs4 import BeautifulSoup
import ssl
import xml.etree.ElementTree as ET
import json

def socket_test():
    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # https use port 443/ http use port 80
    mysock.connect(('mirrors.163.com', 443))
    # mysock.connect(('www.dr-chuck.com', 80))
    

    cmd = 'GET https://mirrors.163.com/.help/fedora-163.repo HTTP/1.1\r\n\r\n'.encode()
    # cmd = 'GET http://www.dr-chuck.com/page1.htm HTTP/1.1\r\n\r\n'.encode()
    mysock.send(cmd)

    while True:
        data = mysock.recv(512)
        if (len(data) < 1):
            break
        print(data.decode())
    mysock.close()

class urllib_test(object):
    def line_print(self):
        fhand = urllib.request.urlopen('https://mirrors.163.com/.help/fedora-163.repo')
        for line in fhand:
            print (line.decode().strip())

    def word_count(self):
        fhand = urllib.request.urlopen('https://mirrors.163.com/.help/fedora-163.repo')
        # for line in fhand:
        #     print (line.decode().strip())

        counts = dict()
        for line in fhand:
            words = line.decode().split()
            for word in words:
                counts[word] = counts.get(word, 0) + 1
        print (counts)
    
    def read_web_page(self):
        fhand = urllib.request.urlopen('http://www.dr-chuck.com/page1.htm')

        # read through the web page. 
        for line in fhand:
            print (line.decode().strip())
        
        # print (fhand.getcode())
        
        ## if you need to get the information in the header. you need to point out which item you want to know. 
        # print (fhand.getheader('ETag'))

    def find_web_link(self):
        fhand = urllib.request.urlopen('http://www.dr-chuck.com/page1.htm')

        # read through the web page. 
        for line in fhand:
            info = line.decode().strip()
            y = re.findall('\"(.+)\"', info)
            if y:
                for i in y:
                    # print (i)
                    f = urllib.request.urlopen(i)
                    for l in f:
                        
                        print (l.decode().strip())
    
    def try_soup(self):
        # with soup to get the link information inside one web page. 
        
        # url = 'https://mirrors.163.com/.help'
        url = 'https://www.amazon.cn/'
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        tags = soup('a')
        # print (tags)
        for tag in tags: 
            # print (tag)
            print (tag.get('href', None))
    
def try_soup():
    
    # without this, https won't work
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # url = 'http://www.dr-chuck.com/page1.htm'
    url = 'http://www.dr-chuck.com'
    html = urllib.request.urlopen(url, context = ctx).read()
    soup = BeautifulSoup(html, 'html.parser')

    tags = soup('a')
    for tag in tags:
        print (tag.get('href', None))

def try_parse_xml():
    data = '''
    <person>
        <name>Chuck</name>
        <phone type="intl">
            +1 734 303 4456
        </phone>
        <email hide="yes"/>
    </person>
    '''
    
    tree = ET.fromstring(data)
    # print (tree)
    print ('Name:', tree.find('name').text)
    print ('Attr:', tree.find('email').get('hide'))
    print ('phone:', tree.find('phone').get("type"))
    print ('phone:', tree.find('phone').text)

def try_parse_xml_2():
    input = '''
    <stuff>
        <users>
            <user x="2">
                <id>001</id>
                <name>Chuck</name>
            </user>
            <user x="7">
                <id>009</id>
                <name>Brent</name>
            </user>
        </users>
    </stuff>    
    '''

    stuff = ET.fromstring(input)
    lst = stuff.findall('users/user')
    print('User count:', len(lst))

    for item in lst:
        print ('Name', item.find('name').text)
        print ('Id', item.find('id').text)
        print ('Attribute', item.get("x"))

def try_json():
    data = '''{
        "name":"Chuck",
        "phone":{
            "type":"intl",
            "number":"+1 724 303 4456"
        },
        "email":{
            "hide":"yes"
        }
    }
    '''

    info = json.loads(data)
    print ('Name:', info["name"])
    print ('Hide:', info["email"]["hide"])
    print ('phone type', info['phone'])
    print ('phone type', info['phone']['type'])

def try_json_2():
    input_info = '''[
        {"id": "001", 
        "x": "2",
        "name": "Chuck"
        },
        {"id": "009", 
        "x": "7",
        "name": "Chuck"
        }
    ]'''

    info = json.loads(input_info)
    print ('User count:', len(info))
    for item in info: 
        print ('Name', item['name'])
        print ('Id', item['id'])
        print ('Attribute', item['x'])

def try_web_service():
    serviceurl = 'http://maps.googleapi.com/maps/api/geocode/json?'

    while True: 
        address = input('Enter location: ')
        if len(address) < 1: break
        
        url = serviceurl + urllib.parse.urlencode({'address': address})
        
        print ('Retrieving', url)
        uh = urllib.request.urlopen(url)
        data = uh.read().decode()
        print ('Retrieved', len(data), 'characters')

        try: 
            js = json.loads(data)
        except: 
            js = None
        if not js or 'status' not in js or js['status'] != 'OK':
            print ('==== Failure To Retrieve ====')
            print (data)
            continue
        
        print (json.dumps(js, indent=4))

        lat = js["results"][0]["geometry"]["location"]["lat"]
        lng = js["results"][0]["geometry"]["location"]["lng"]

        print ('lat', lat, 'lng', lng)
        location = js['results']
        print (location)


if __name__ == '__main__':
    # a = urllib_test()
    # a.try_soup()
    # socket_test()

    try_web_service()