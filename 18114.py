#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader
import datetime
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm  import SVR
from sklearn import preprocessing


# In[2]:


df=pd.read_csv('/home/jemish/Desktop/ML Project/Nifty/NIFTY 50.csv')
df=pd.DataFrame(df)
print(df)
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
index = df['Date']
df.plot(x = 'Date', y = 'Close',label='Close price',title='NIFTY 50')
plt.legend()
plt.savefig('Nifty_2_1.png', bbox_inches='tight')
plt.show()


# # Naive Bayes

# In[3]:


from sklearn.metrics import plot_confusion_matrix as pcm
Daily_return=np.zeros(len(df))
classification=np.zeros(len(df))

for i in range(1,len(df)):
    Daily_return[i]=(df['Close'][i]-df['Close'][i-1])
for i in range(0,len(df)-1):
    if Daily_return[i+1]<0:
        classification[i]=0
    else:
        classification[i]=1

df['Daily Return']=Daily_return
df['Classification']=classification
x1=df['Daily Return']  #df['Close']
y1=df['Classification']
x=np.zeros((len(df),1))
y=np.zeros((len(df),1))
for i in range(len(df)):
    x[i][0]=x1[i]
    y[i][0]=y1[i]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,shuffle=False)
model=GaussianNB()
model.fit(x_train,y_train)
y_prediction=model.predict(x_test)
cm=confusion_matrix(y_test,y_prediction)
print("Confusion Matrix")
print(cm)
print(classification_report(y_test,y_prediction))
print(f"F1 Score:{f1_score(y_test,y_prediction)}")
accuracy=accuracy_score(y_test,y_prediction)
print("Accuracy : %f"%(accuracy*100))
pcm(model,x_test,y_test)
plt.savefig('Cofusion_matrix.png', bbox_inches='tight')
plt.show()


# # Linear Regression

# In[4]:



x=np.array(df.index).reshape(-1,1)
y=df['Close']

linreg=LinearRegression().fit(x,y)
a=linreg.score(x,y)
predictions=linreg.predict(x)
#plt.figure(figsize=(15,5))
plt.scatter(x,df['Close'],s=1,label='Close Price')
plt.plot(df.index,predictions,label='Prediction',color='orange')
plt.legend()
plt.savefig('linear_regression_3_1.png', bbox_inches='tight')
plt.show()
print("Slope:",np.asscalar(np.squeeze(linreg.coef_)))
print("Intercept:",linreg.intercept_)
print('R^2:',linreg.score(x,y))


# # Logistic Regression
We are using 10 day moving average,correlation,RSI(Relative Strength Index),difference between two successive day open,day close,high,low and close price as indicators to make the predication. 
# In[9]:



df['S_10']=df['Close'].rolling(window=10).mean()
df['Corr']=df['Close'].rolling(window=10).corr(df['S_10'])
df['Open-Close']=df['Open']-df['Close'].shift(1)
df['Open-Open']=df['Open']-df['Open'].shift(1)
df1=df.drop(['Date','Volume','Turnover','P/E','P/B','Div Yield','Daily Return','Classification'],axis=1)
x=df1.iloc[18:,:8]
print(x)

Define Target:
The dependent variable is the same as discussed in the above example. If tomorrow's closing price is higher than today'sclosing price,then we will buy the stock(1),else we will sell it(-1)
# In[10]:


y=np.where(df1['Close'].shift(-1)>df1['Close'],1,-1)
split=int(0.7*len(x))
y=y[18:]
x_train,x_test,y_train,y_test=x[:split],x[split:],y[:split],y[split:]
model=LogisticRegression()
model=model.fit(x_train,y_train)
pd.DataFrame(zip(x.columns,np.transpose(model.coef_)))


# In[11]:


probability=model.predict_proba(x_test)
predicted=model.predict(x_test)
print("Confusion Matrix:\n",confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))
print("Model Accuracy:",model.score(x_test,y_test)*100)
cross_val=cross_val_score(LogisticRegression(),x,y,scoring='accuracy')
print("Accuracy after cross validation:",cross_val.mean()*100)
print(f"F1 Score:{f1_score(y_test,predicted)}")


# In[12]:



df2=pd.DataFrame()
df2['Predicted_Signal']=model.predict(x)
df2['Nifty_returns']=np.log(df1['Close']/df1['Close'].shift(1))
cumulative_Nifty_return=np.cumsum(df2[split:]['Nifty_returns'])
df2['Strategy_returns']=df2['Nifty_returns']*df2['Predicted_Signal'].shift(1)
cumulative_Strategy_return=np.cumsum(df2[split:]['Strategy_returns'])
print(len(df2[split:]))
plt.plot(cumulative_Nifty_return,color='r',label='Nifty Returns')
plt.plot(cumulative_Strategy_return,color='g',label='Strategy Returns')
plt.xlabel('last 1558 days from 4th December 2020 ')
plt.ylabel('Returns (%)')
plt.legend()
plt.savefig('logistic_regression_2_1.png', bbox_inches='tight')
plt.show()


# # SVM

# In[13]:


df3=df[['Close']]
df3['Prediction']=df3[['Close']].shift(-150)
x=np.array(df3.drop(['Prediction'],1))
x=x[:-150]
y=np.array(df3['Prediction'])
y=y[:-150]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
svr=SVR(kernel='rbf',C=1e5,gamma=0.000001)
svr.fit(x_train,y_train)
svm_confidence=svr.score(x_test,y_test)
print("SVM Confidence:",svm_confidence)
forecast=np.array(df3.drop(['Prediction'],1))[-150:]
pred=svr.predict(forecast)
#plt.figure(figsize=(10,5))
plt.plot(forecast,label="Last 150 Days Nifty Close Price")
plt.plot(pred,label="SVM Prediction")
plt.ylabel('Index price')
plt.xlabel('Last 150 days from 4th december 2020')
plt.legend()
plt.savefig('SVM_6_ge-6_1', bbox_inches='tight')
plt.show()


# # Random Forest
Input:(Open-Close)/Open,(High-Low)/High,standard deviation of last 5 days(std-5),average of last 5 days returns(ret_5)
Output: if tommorrow's close price is higher than today's close price than signal to buy stock (1) otherwise sell stock(-1)
# In[29]:


from sklearn.ensemble import RandomForestClassifier
df4=pd.DataFrame(df)
df4['Open-Close']=(df4.Open-df4.Close)/df4.Open
df4['High-Low']=(df4.High-df4.Low)/df4.Low
df4['Percent_Change']=df4['Close'].pct_change()
df4['Std_5']=df4['Percent_Change'].rolling(5).std()
df4['Ret_5']=df4['Percent_Change'].rolling(5).mean()
df4.dropna(inplace=True)

x=df4[['Open-Close','High-Low','Std_5','Ret_5']]
y=np.where(df4['Close'].shift(-1)>df4['Close'],1,-1)

length=df4.shape[0]
split=int(length*0.75)
x_train,x_test,y_train,y_test=x[:split],x[split:],y[:split],y[split:]

clf=RandomForestClassifier(random_state=5)
model=clf.fit(x_train,y_train)
pred=model.predict(x_test)
print('Correct Prediction (%):',accuracy_score(y_test,pred,normalize=True)*100)
print(classification_report(y_test,pred))
print(f"F1 Score:{f1_score(y_test,pred)}")
df4['strategy_return']=df4.Percent_Change.shift(-1)*model.predict(x)
(df4.strategy_return[split:]+1).cumprod().plot(label="Last 1288 Days Return By Strategy")
plt.plot(cumulative_Nifty_return,color='r',label='Nifty Returns')
plt.ylabel('Strategy Return(%)')
plt.legend()
plt.savefig('RandomForest_return_1.png', bbox_inches='tight')
plt.show()


# # ANN

# In[36]:


from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


# In[37]:


df5=df[['Close','Open']]
df5['Prediction']=df5['Open'].shift(-1)
sc=MinMaxScaler(feature_range=(0,1))
df5=sc.fit_transform(df5)
df5=pd.DataFrame(df5)
x=df5.iloc[:,0]
y=df5.iloc[:,2]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,shuffle=False)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(600, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer="adam", loss="mean_squared_error",metrics = ["accuracy"])

model.fit(x_train, y_train,epochs=10)
y_pred=model.predict(x_test)
a=len(y_train)
y1=np.zeros(len(y_test))
for i in range(a,len(y_test)+a):
    y1[i-a]=y_test[i]
y2=np.zeros(shape=(len(y1),3))
for i in range(len(y1)):
    y2[i][0]=y1[i]
y2=sc.inverse_transform(y2)
q=y2[:,0]
y4=np.zeros(shape=(len(y_pred),3))
for i in range(len(y1)):
    y4[i][0]=y_pred[i][0]
y4=sc.inverse_transform(y4)
p=y4[:,0]
plt.plot(q,label="Nifty 50")
plt.plot(p,label="Prediction of last 782 days")
plt.xlabel('last 782 days from 4th December 2020')
plt.legend()
plt.savefig("ANN_Prediction_1_1.png",bbox_inches='tight')
plt.show()


# # Unsupervised Learning: PCA

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df4=pd.DataFrame(df)
df4['Open-Close']=(df4.Open-df4.Close)/df4.Open
df4['High-Low']=(df4.High-df4.Low)/df4.Low
df4['Percent_Change']=df4['Close'].pct_change()
df4['Std_5']=df4['Percent_Change'].rolling(5).std()
df4['Ret_5']=df4['Percent_Change'].rolling(5).mean()
df4.dropna(inplace=True)

x=df4[['Open-Close','High-Low','Std_5','Ret_5']]
y=np.where(df4['Close'].shift(-1)>df4['Close'],1,-1)

length=df4.shape[0]
split=int(length*0.75)
x_train,x_test,y_train,y_test=x[:split],x[split:],y[:split],y[split:]
pca=PCA(n_components=4)
X_train=pca.fit_transform(x_train)
X_test=pca.transform(x_test)
explain_variance= pca.explained_variance_ratio_
print(f"Explain Variance Ratios:{explain_variance}")
clf=RandomForestClassifier(random_state=5)
model=clf.fit(X_train,y_train)
pred=model.predict(X_test)
print('Correct Prediction (%):',accuracy_score(y_test,pred,normalize=True)*100)
print(classification_report(y_test,pred))

print(f"F1 Score:{f1_score(y_test,pred)}")
df4['strategy_return']=df4.Percent_Change.shift(-1)*model.predict(x)
(df4.strategy_return[split:]+1).cumprod().plot(label="Last 1298 Days Return By Strategy")
plt.plot(cumulative_Nifty_return,color='r',label='Nifty Returns')
plt.ylabel('Strategy Return(%)')
plt.legend()
plt.savefig('PCA.png', bbox_inches='tight')
plt.show()


# In[ ]:




