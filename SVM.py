#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[5]:


cell_df = pd.read_csv('cell_samples.csv')
print(cell_df.tail(5))


# In[11]:


cell_df['Class'].value_counts()


# In[21]:


ax=cell_df[cell_df['Class']==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='DarkBlue',label='malignant');

cell_df[cell_df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='Yellow',label='benign',ax=ax);


# In[23]:


cell_df.dtypes


# In[24]:


cell_df['BareNuc'].value_counts()


# In[32]:


cell_df=cell_df[pd.to_numeric(cell_df['BareNuc'],errors='coerce').notnull()]
cell_df['BareNuc']=cell_df['BareNuc'].astype('int')
print(cell_df['BareNuc'].value_counts())
print(cell_df.dtypes)
print(cell_df.columns)


# In[42]:


feature_df=cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
x=np.asarray(feature_df)
print(x[0:5])


# In[40]:


y=np.asarray(cell_df['Class'])
print(y[0:10])


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print('Train set:',x_train.shape,y_train.shape)
print('Test set:',x_test.shape,y_test.shape)


# In[46]:


#### Modeling


# In[50]:


from sklearn import svm
clf=svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)


# In[52]:


#### Predecting


# In[54]:


yhat=clf.predict(x_test)
yhat[0:5]


# In[56]:


#### Evaluation


# In[57]:


from sklearn.metrics import classification_report,confusion_matrix
import itertools
def plot_confusion_matrix(cm,classes,normalize=False,title='confusion matrix',cmap=plt.cm.Blues):
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    print(confusion_matrix(y_test, yhat, labels=[1, 0]))


# In[60]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


# In[61]:


##f1_score


# In[62]:


from sklearn.metrics import f1_score
f1_score(y_test,yhat,average='weighted')


# In[63]:


####Jaccard_score


# In[68]:


from sklearn.metrics import jaccard_score
print('jaccard_scor 2:',jaccard_score(y_test,yhat,pos_label=2))
print('jaccard_scor 4:',jaccard_score(y_test,yhat,pos_label=4))

