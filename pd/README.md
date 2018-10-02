# pandas daily usage

## package hierarchy

- `__init__.py` : inherit package name 

## Note: 
### Creation:

- __np__ create array: 
    - ex (create rows and cols):
        - np.array([[1, 2], [3, 4], [5, 6]])

### Selection:
- __pd__ .loc is to choose the label things, 
    - ex (select specific row and col): 
        - pos = y == 0;
        - df.loc[pos, df.columns[0]] 
- __np__ compare np selection
    - ex (select one col): 
        - test = np.array([[1, 2], [3, 4], [5, 6]])
        - test[:,0]



### Plot:
- 2D scatter plot with different legend