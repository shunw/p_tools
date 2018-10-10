# pandas daily usage

## package hierarchy

- `__init__.py` : inherit package name 

## Note: 
### Differentiate dataframe and matrix:
- __np__: if type(X) is np.ndarray
- __pd__: if type(X) is pd.DataFrame

### Creation:

- __np__ create array: 
    - ex (create rows and cols):
        - np.array([[1, 2], [3, 4], [5, 6]])

### Selection:
- __pd__ .loc is to choose the label things, 
    - ex (select specific row and col): 
        - pos = y == 0;
        - df.loc[pos, df.columns[0]] 
    - ex2: 
        - pos = y == 0;
        - plt.scatter(X.loc[pos, X.columns[0]], X.loc[pos, X.columns[1]], marker = '^')
- __np__ compare np selection
    - ex (select one col): 
        - test = np.array([[1, 2], [3, 4], [5, 6]])
        - test[:,0]
    - ex2: 
        - pos = y == 0;
        - plt.scatter(X[pos, 0], X[pos, 1], marker = '^')



### Plot:
- 2D scatter plot with different legend
    - ex:
        - ax1 = plt.scatter(x0, x1, marker = '^')
        - ax2 = plt.scatter(x0, x1, marker = '^')
        - plt.legend((ax1, ax2), ('admitted', 'not admitted'), loc = 'lower left')