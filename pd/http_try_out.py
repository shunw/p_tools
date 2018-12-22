###############
# @author sw
# @date 2018-09-28
#######################

import socket
import urllib.request, urllib.parse, urllib.error

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
            print (line.decode().strip())
    


if __name__ == '__main__':
    a = urllib_test()
    a.read_web_page()
    # socket_test()