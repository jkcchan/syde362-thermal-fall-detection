clf=linear_model.LogisticRegression(verbose=10, tol=0.000000000000001)
data = pd.read_csv('dataset.csv')
data = data.reset_index()[['falling','area','bb_ratio','hu1','hu2','hu3','hu4','hu5','hu6','hu7','variance']]
y = data.falling
X = data.drop(['falling'], axis=1)
scaler = MinMaxScaler()
X_scaled = scaler.fit(X)
X_scaled = scaler.transform(X)
X = X_scaled
X_train, X_test, y_train, y_test = train_test_split(X, y)
logit = dm.Logit(y_train, X_train)
f = []
def sigmoid(x):
    return 1/(1+np.power(np.e, -x))
h = lambda theta, x:sigmoid(x.dot(theta))
def cost(theta, X, y, lambd=0, debug=False, **kwargs):
    """ Logistic regression cost function with optional regularization. Lambd is the regularization constant. """
    m = X.shape[0]
    j = y.dot(np.log(h(theta, X)))  + (1 - y).dot(np.log(1 - h(theta, X)))
    regularization = (float(lambd)/float(2*m)) * theta[1:].dot(theta[1:].T)
    j /= -m
    j += regularization
    return j
def my_callback(xi):
    a =  xi.tolist()
    c = cost(xi, X_test, y_test)
    a.append(c)
#     print a
    f.append(a)
a = logit.fit(callback=my_callback)
#a.params