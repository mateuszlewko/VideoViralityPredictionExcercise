from math import log, e
from sklearn import linear_model
from random import shuffle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plot


FILE_NAME = 'data.csv'
LAST = 167
X_CNT = 24

data = []

# parse data #
with open(FILE_NAME) as file:
    for row in file:
        data.append([float(val) for val in row.split(',')[1:]])
        
shuffle(data)
TRAIN_CNT = int(.9 * len(data))
print 'train cnt:', TRAIN_CNT

Y = [item[LAST] for item in data]
Y_log = map(lambda x: log(x), Y)
X = []

X_lr = []
X_lr_log = []

# compute X_lr and X_lr_log #
for item in data:
    regr = linear_model.LinearRegression()
    regr.fit([[val] for val in range(1, X_CNT + 1)], item[:X_CNT])
    
    X_lr.append(regr.coef_[0])
    X_lr_log.append(log(regr.coef_[0]))
    
#print Y_log

# train LinearRegression model #
train_regr = linear_model.LinearRegression()
train_regr.fit([[val] for val in X_lr_log[:TRAIN_CNT]], Y_log[:TRAIN_CNT])


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
    

def plot_LinearRegression():
    """ plot LinearRegression coeff of first 24 hours and Output (Y - original)
    """
    plot.scatter(X_lr, Y)
    plot.show()

    # log transformed
    plot.title('X - normal, Y - log')
    plot.scatter(X_lr, Y_log)
    plot.show()

    # both log transformed
    plot.title('X - log, Y - log')
    plot.scatter(X_lr_log, Y_log)
    plot.show()


    train_regr = linear_model.LinearRegression()
    train_regr.fit([[val] for val in X_lr_log[:TRAIN_CNT]], Y_log[:TRAIN_CNT])

def predict():
    Y_predicted = []
    error_sum = 0

    for coeff, real in zip(X_lr_log[TRAIN_CNT:], Y[TRAIN_CNT:]):
        pred = e ** train_regr.predict(coeff)[0]
        
        print 'real:', '%9s' % str(int(real)), '| predicted:', int(pred)
        
        error_sum += abs(float(pred) / real - 1.0) ** 2
        Y_predicted.append(pred)
        
    print '\nmRSE:', error_sum / float(len(Y_predicted))
    
    
if __name__ == '__main__':
    plot_distribution()
    plot_LinearRegression()
    
    predict()