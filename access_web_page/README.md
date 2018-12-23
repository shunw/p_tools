## Access Web Data

### Sockets in Python
- built-in support for TCP Sockets
    ```py
    # setup the connection. 
    import socket
    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysock.connect(('data.pr4e.org', 80))
    ```

- An HTTP request in python
    ```py
    import socket

    mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysock.connect(('data.pr4e.org', 80))

    cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
    mysock.send(cmd)

    while True:
        data = mysock.recv(512)
        if (len(data) < 1):
            break
        print(data.decode())
    mysock.close()
    ```

### urllib in python
    ```py
    import urllib.request, urllib.parse, urllib.error

    fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
    for line in fhand:
        print(line.decode().strip())
    ```

### use soup to find the link
    ```py
    url = somelink
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    
    tags = soup('a')
    # print (tags)
    for tag in tags: 
        # print (tag)
        print (tag.get('href', None))
    ```
