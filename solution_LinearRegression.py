from math import log, e
import matplotlib.pyplot as plot


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
 
#print Ylog

for day in [6, 24, 72, 168]:
    d = [item[day - 1] for item in data]
    
    plot.hist(d, bins=250)
    plot.title('Original, up to day: ' + str(day))
    plot.show()
    
    plot.hist(map(lambda x: log(x), d), bins=250)
    plot.title('Log(x), up to day: ' + str(day))
    plot.show()
