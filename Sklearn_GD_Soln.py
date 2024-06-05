import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from libs.ML_Course_libs.lab_utils_multi  import load_house_data
from libs.ML_Course_libs.lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('/Users/Demian/Desktop/pythonProject/libs/ML_Course_libs/deeplearning.mplstyle')

#Load the dataset
X_train, y_train=load_house_data()
X_feature=['size(sqft)', 'bedrooms', 'floors','age']

#Scale/normalize the training data
scaler=StandardScaler()
X_norm=scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X :{np.ptp(X_train, axis=0)} ")
print(f"Peak to Peak range by column in Normalized X :{np.ptp(X_norm, axis=0)}")

#Create and fit the regression model
sgdr=SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed : {sgdr.n_iter_}, number of weight_update: {sgdr.t_}")
#View parameters
b_norm=sgdr.intercept_
w_norm=sgdr.coef_
print(f"model parameters:                          w:{w_norm}, b:{b_norm}")
print(f"model parameters from the previous lab     w:[110.56 -21.27 -32.71 -37.97], b:[363.16]")

#Make predictions
#make a prediction using sgdr.predict()
y_pred_sgd=sgdr.predict(X_norm)
#make a prediction using w,b
y_pred=np.dot(X_norm, w_norm)+b_norm

print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:10]}" )
print(f"Target values \n{y_train[:10]}")
print(y_pred)
print(y_train)

fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train, label='target')
    ax[i].set_xlabel(X_feature[i])
    ax[i].scatter(X_train[:,i], y_pred, color=dlc["dlorange"], label='predict')
ax[0].set_ylabel("Price"); ax[0].legend()
fig.suptitle("target vs prediction using z-score normalized model")
plt.show()

