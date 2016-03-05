from math import log, e
from sklearn import linear_model
from random import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot


FILE_NAME = 'data.csv'
LAST = 167
X_CNT = 24

data = []

with open(FILE_NAME) as file:
    for row in file:
        data.append([float(val) for val in row.split(',')[1:]])
        
shuffle(data)
TRAIN_CNT = int(.9 * len(data))
        
Y = [item[LAST] for item in data]
Y_log = map(lambda x: log(x), Y)
X = []
 
#print Y_log

def plot_distribution():
    """Distribution for diferrent days and log transformed
    """
    for day in [6, 24, 72, 168]:
        d = [item[day - 1] for item in data]
        
        plot.hist(d, bins=250)
        plot.title('Original, up to day: ' + str(day))
        plot.show()
        
        plot.hist(map(lambda x: log(x), d), bins=250)
        plot.title('Log(x), up to day: ' + str(day))
        plot.show()
    
X_lr = []
for item in data:
    regr = linear_model.LinearRegression()
    regr.fit([[val] for val in range(1, X_CNT + 1)], item[:X_CNT])
    
    X_lr.append(regr.coef_[0])
    
# plot LinearRegression coeff of first 24 hours and Output (Y - original)
plot.scatter(X_lr, Y)
plot.show()

# log transformed
plot.scatter(X_lr, Y_log)
plot.show()

print 'train cnt:', TRAIN_CNT

train_regr = linear_model.LinearRegression()
train_regr.fit([[val] for val in X_lr[:TRAIN_CNT]], Y[:TRAIN_CNT])

Y_predicted = []
error_sum = 0

for coeff, real in zip(X_lr[TRAIN_CNT:], Y[TRAIN_CNT:]):
    pred = train_regr.predict(coeff)
    print 'real:', real, '| predicted:', pred[0]
    
    error_sum += (float(pred) / real - 1.0) * (float(pred) / real - 1.0)
    Y_predicted.append(pred)
    
# something wron with skl msr
print 'error?:', mean_squared_error(Y[TRAIN_CNT:], Y_predicted)
print 'mRSE:', error_sum / float(len(Y_predicted))