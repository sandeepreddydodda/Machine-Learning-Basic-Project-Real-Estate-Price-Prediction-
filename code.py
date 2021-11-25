import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

dict = {'area':[2600, 3000, 3200, 3600, 4000], 'price':[550000, 565000, 610000, 680000, 725000]}
df = pd.DataFrame(dict)
df

%matplotlib inline
plt.scatter(df.area, df.price, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
plt.xlabel('area')
plt.ylabel('price')
plt.plot(df.area, reg.predict(df[['area']]), color='red')

reg.predict([[3300]])
reg.coef_
reg.intercept_
