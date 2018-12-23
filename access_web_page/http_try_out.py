###############
# @author sw
# @date 2018-09-28
#######################

import socket
import urllib.request, urllib.parse, urllib.error
import re
from bs4 import BeautifulSoup
import ssl

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


if __name__ == '__main__':
    # a = urllib_test()
    # a.try_soup()
    # socket_test()

    try_soup()