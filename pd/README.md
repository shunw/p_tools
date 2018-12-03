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
        ```
        np.array([[1, 2], [3, 4], [5, 6]])

### Selection:
- __pd__ .loc is to choose the label things, 
    - ex (select specific row and col): 
        ``` 
        pos = y == 0;
        df.loc[pos, df.columns[0]] 
    - ex2: 
        ``` 
        pos = y == 0;
        plt.scatter(X.loc[pos, X.columns[0]], X.loc[pos, X.columns[1]], marker = '^')
- __np__ compare np selection
    - ex (select one col): 
        ```
        test = np.array([[1, 2], [3, 4], [5, 6]])
        test[:,0]
    - ex2: 
        ``` 
        pos = y == 0; 
        plt.scatter(X[pos, 0], X[pos, 1], marker = '^')
        note: the pos here should be 1D data. If it's 2D data, need to flatten it. 



## Plot:
### Legend
- 2D scatter plot with different legend
    ``` 
    ax1 = plt.scatter(x0, x1, marker = '^')
    ax2 = plt.scatter(x0, x1, marker = '^')
    plt.legend((ax1, ax2), ('admitted', 'not admitted'), loc = 'lower left')
- plot with legend
    ```
    err_trn, = plt.plot(x1, y1, label = 'Train')
    err_val,  = plt.plot(x1, y2, label = 'Cross Validation')
    plt.legend(handles = [err_trn, err_val])
### Separate Plot    
- plot 2 separate plot
    ```
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
    ```
    fig = plt.figure()
    _ = plt.scatter(X[pos, 0], X[pos, 1])
    _ = plt.scatter(X[neg, 0], X[neg, 1])
    return fig
- call func
    ```
    f = function(...)
    plt.plot(X, y)
    plt.show()    

### Color Map
- ex1: 
    ```
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for i, c in enumerate(colors):
    plt.scatter(X[idx == i, 0], X[idx == i, 1], c = 'white', edgecolors = c, s = 15)
    
- ex2: 
    ```
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / K) for i in range(K)]
    colors = np.array(colors)

## Regular Expression