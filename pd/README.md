# pandas/numpy daily usage

## package hierarchy

- `__init__.py` : inherit package name 

## Data Deal: 
### Differentiate dataframe and matrix:
- __np__: if type(X) is np.ndarray
- __pd__: if type(X) is pd.DataFrame

### Creation:

- __np__ create array: 
    - ex (create rows and cols):
        ```py
        np.array([[1, 2], [3, 4], [5, 6]])

### Selection:
- __pd__ .loc is to choose the label things, 
    - ex (select specific row and col): 
        ```py 
        pos = y == 0
        df.loc[pos, df.columns[0]] 
    - ex2: 
        ```py 
        pos = y == 0
        plt.scatter(X.loc[pos, X.columns[0]], X.loc[pos, X.columns[1]], marker = '^')

- __np__ compare np selection
    - ex (select one col): 
        ```py
        test = np.array([[1, 2], [3, 4], [5, 6]])
        test[:,0]
    - ex2: 
        ```py 
        pos = y == 0
        plt.scatter(X[pos, 0], X[pos, 1], marker = '^')
        # note: the pos here should be 1D data. If it's 2D data, need to flatten it. 



## Plot:
### Legend
- 2D scatter plot with different legend
    ```py 
    ax1 = plt.scatter(x0, x1, marker = '^')
    ax2 = plt.scatter(x0, x1, marker = '^')
    plt.legend((ax1, ax2), ('admitted', 'not admitted'), loc = 'lower left')
- plot with legend
    ```py
    err_trn, = plt.plot(x1, y1, label = 'Train')
    err_val,  = plt.plot(x1, y2, label = 'Cross Validation')
    plt.legend(handles = [err_trn, err_val])
### Separate Plot    
- plot 2 separate plot
    ```py
    plt.figure(1)
    plt.subplot(211)
    plt.scatter ...
    plt.subplot(212)
    plt.hist ...

    plt.figure(2)
    plt.plot ...

    plt.show()

### Return Plot by Func
- within func
    ```py
    fig = plt.figure()
    _ = plt.scatter(X[pos, 0], X[pos, 1])
    _ = plt.scatter(X[neg, 0], X[neg, 1])
    return fig

- call func
    ```py
    f = function(...)
    plt.plot(X, y)
    plt.show()    

### Color Map
- ex1: 
    ```py
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for i, c in enumerate(colors):
    plt.scatter(X[idx == i, 0], X[idx == i, 1], c = 'white', edgecolors = c, s = 15)
    
- ex2: 
    ```py
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / K) for i in range(K)]
    colors = np.array(colors)

## Regular Expression
### Quick Guide
Symbol | Quick Guide | Example
--- | --- | ---
^ | Matches the `beginning` of a line | `^X.*:` => `X-Sieve:` CMU Sieve 2.3
$ | Matches the `end` of the line | ---
. | Matches `any` character | ---
\s | Matches `whitespace` | ---
\S | Matches any `non-whitespace` character | `^X-\S+:` !=>`X-Plane is behind schedule:` two weeks
* | `Repeats` a character zero or more times | ---
*? | `Repeats` a character zero or more times (non-greedy) | ---
+ | `Repeats` a character one or more times | ---
+? | `Repeats` a character one or more times (non-greedy) | ---
[aeiou] | Matches a single character in the listed `set` | ---
[^XYZ] | Matches a single character `not in` the listed `set` | ---
[a-z0-9] | The set of characters can include a `range` | ---
( | Indicates where string `extraction` is to start | ---
) | Indicates where string `extraction` is to end | ---

- re.search()
    - returns True or False
    ```py
    if re.search('^Frome:', line):
        print (line)

- re.findall()
    - want the matching strings to be extracted
    ```py
    x = 'My 2 favorite numbers are 19 and 42'
    y = re.findall('[0-9]+', x)
    print (y)
    # result is ['2', '19', '42']

### Greedy/ Non-Greedy Matching
- Greedy: * and + push `outward` in both directions to match the largest possible string
    ```py
    x = 'From: Using the : character'
    y = re.findall('^F.+:', x)
    print (y)
    # output is: ['From: Using the :']

- Non-Greedy: add a ? character, the + and * chill out a bit
    ```py
    x = 'From: Using the :character'
    y = re.findall('^F.+?:', x)
    print (y)
    # output is: ['From:']

### String Extraction
- Parentheses are not part of the match, but they tell where to `start` and `stop` what string to extract
    - ex1
        ```py
        x = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
        y = re.findall('\S+@\S+', x)
        print (y)
        # output is: ["stephen.marquard@uct.ac.za"]

        y = re.findall('^From (\S+@\S+)', x)
        print (y)
        # output is ["stephen.marquard@uct.ac.za"]

    - ex2 
        ```py
        x = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
        y = re.findall('@([^ ]*)', x) # [^ ]match non-blank character
        print (y)
        # output is ['uct.ac.za']

### Escape Character
- want a special regular expression character to just behave `normally`, you prefix it with '\'
    ```py
    x = 'We just received $10.00 for cookies.'
    y = re.findall('\$[0-9.]+', x)
    print (y)
    # output is ['$10.00']

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