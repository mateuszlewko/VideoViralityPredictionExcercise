from math import log, e

FILE_NAME = 'data.csv'
LAST = 167
IN_CNT = 24

data = []

with open(FILE_NAME) as file:
    for row in file:
        data.append([float(val) for val in row.split(',')[1:]])
        
LEARN_CNT = int(.9 * len(data))
        
Y = [item[LAST] for item in data]
Ylog = map(lambda x: log(x), Y)
X = []
 
print Ylog

 