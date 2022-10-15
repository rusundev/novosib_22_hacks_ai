#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import recall_score, precision_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


IN_DIR = r'E:\hacks-ai\2022.10.Новосибирская область\in'


# In[86]:


#df_train = pd.read_csv("/content/train.csv")

df_train = pd.read_csv(IN_DIR + "\\train_dataset_train.csv")


# ## Рассмотрим датасет по ближе

# In[4]:


df_train.shape


# In[5]:


df_train.info()


# # Пояснение к данным
# ## Столбец «Class» хранит в себе тип класса точки, где:
# ### 0 – точки земли
# ### 1 – точки опор
# ### 3 – точки растительности
# ### 4 – точки рельсов
# ### 5 – точки элементов контактной сети
# ### 64 – точки шумов

# In[40]:


# Коды классов для именования подмножеств данных
class_dict = {
    'ground': 0,
    'support': 1,
    'green': 3,
    'rails': 4,
    'wires': 5,
    'noise': 64
}


# Пострим на распределение данных

# In[6]:


sns.countplot(x = "Class" , data  = df_train).set_title('Распределение класса точки')


# In[17]:


sns.set(rc={'figure.figsize':(18,10)})
sns.stripplot(data = df_train, x= "Class", y = "Reflectance").set_title('Зависимость класса от параметра отражения');


# In[30]:


sns.set(rc={'figure.figsize':(18,10)})
sns.stripplot(data = df_train, x= "Class", y = "Height").set_title('Зависимость класса от высоты');


# In[32]:


sns.set(rc={'figure.figsize':(18,10)})
sns.stripplot(data = df_train, x= "Class", y = "Easting").set_title('Зависимость класса от направления на восток');


# In[31]:


sns.set(rc={'figure.figsize':(18,10)})
sns.stripplot(data = df_train, x= "Class", y = "Northing").set_title('Зависимость класса от направления на север');


# In[34]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_train, x= "Easting", y = "Northing").set_title('Точки по координатам Восток, Север');


# In[35]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_train, x= "Easting", y = "Height").set_title('Точки по координатам Восток, Высота');


# In[58]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_train[~df_train['Class'].isin([0,3])], x= "Northing", y = "Height", hue="Class").set_title('Точки по координатам Север, Высота');


# In[59]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_train[~df_train['Class'].isin([0,3])], x= "Easting", y = "Height", hue="Class").set_title('Точки по координатам Восток, Высота');


# In[ ]:





# In[69]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

data_3d = df_train[~df_train['Class'].isin([0, 3, 64])]

ax.scatter(data_3d['Northing'], data_3d['Easting'], data_3d['Height'], c=data_3d['Class'])

plt.title('3D визуализация точек ЖД-инфраструктуры с отсеенным фоном')

ax.view_init(30, 120)

plt.show()


# In[ ]:





# In[53]:


df_train.groupby(['Class']).count()


# In[57]:


df_train[~df_train['Class'].isin([0,3])]


# In[18]:


plt.rcParams['figure.figsize']=(15,15)

g = sns.heatmap(df_train.corr(), square = True, annot=True)


# ## Выделим выборки

# In[6]:


df_train = df_train.fillna(0)


# In[7]:


X = df_train.drop(["Class", "id"], axis = 1)
y = df_train[["Class"]]


# In[8]:


X.shape


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[10]:


y_train


# ## Обучение модели

# In[24]:


clf = RandomForestClassifier(random_state=0)


# In[ ]:


clf.fit(X_train, y_train)


# ## Оценка точности

# In[ ]:


pred = clf.predict(X_test)


# In[ ]:


y_test.head(3)


# In[ ]:


result = recall_score(y_test, pred, average='macro', zero_division=True)

print("Recall score",result)


# In[ ]:





# In[11]:


df_test = pd.read_csv(IN_DIR + "\\test_dataset_test.csv")


# In[12]:


df_test.shape


# In[13]:


df_test.info()


# In[43]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_test, x= "Easting", y = "Northing").set_title('Точки по координатам Восток, Север');


# In[44]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_test, x= "Easting", y = "Height").set_title('Точки по координатам Восток, Высота');


# In[45]:


sns.set(rc={'figure.figsize':(18,10)})
sns.scatterplot(data = df_test, x= "Northing", y = "Height").set_title('Точки по координатам Север, Высота');


# In[ ]:





# # Метод KDE (kernel density estimation) оценки распределения точек и их классов
# 
# ## 1. Выделение точек классов
# ### 1.1 Приоритетными считаются точки рельсов, опор и элементов контактной сети (значения "Class": 4, 1, 5)
# ### 1.2 Фон изображения: земля и растительность ("Class": 0, 3) и шумы ("Class": 64) возможно даже имеет смысл отфильтровывать для работы только с информативным сигналом.
# 
# ## 2. Получение KDE отдельно для классов.
# ### 2.1 Выделение подмножеств данных для каждого класса (возможно самый многочисленный для земли можно пропустить).
# ### 2.2 Определение KDE для отдельного класса.
# ### 2.3 Оценка min и max значений ядерной плотности класса. Min будет использован для порогового значения (threshold) при определении класса.
# 
# ## 3. Получение предсказаний классов для тестовых точек. Оценка точности предсказания.

# In[49]:


# Для смещения координат в 0. Меньше обрабатываемые числа - экономия памяти и ресурсов процессора.
dict_min = {col: df_train[col].min() for col in ['Easting', 'Northing', 'Height', 'Reflectance']}
dict_min


# In[87]:


# Конвертация - сдвига к нулю для уменьшения обрабатываемых чисел
df_train_conv = df_train.copy()

# TODO: возможно поворот, чтобы данные были похожи на прямоугольник, улучшит точность

for col in dict_min.keys():
    df_train_conv[col] = df_train_conv[col] - dict_min[col]

df_train_conv.describe()


# In[102]:


import numpy as np
import pandas as pd
import statsmodels.api as sm

"""
Возвращает объект KDE для поданных на вход точек
"""
def get_kde(points_list, var_type='ccc', bw='normal_reference'):
    dens_u = sm.nonparametric.KDEMultivariate(data=points_list, var_type=var_type, bw=bw)
    pdf_points = ''

    return dens_u

"""
Вычисляет значения плотности точек PDF для поданных на вход точек и их PDF
"""
def get_pdf_points(dens_u, points_list):
    pdf_col = []
    for point in points_list:
        pdf_col.append(np.log10(dens_u.pdf(data_predict=point)))
        
    return pdf_col


# In[105]:


points_list = df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]),
                              ['Easting', 'Northing', 'Height']].to_numpy().tolist()

dens_u = get_kde(points_list)


# In[113]:


len(points_list)


# In[119]:


points_list[:10]


# In[115]:


pdf_list = get_pdf_points(dens_u, points_list)


# In[123]:


fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(projection='3d')
ax.scatter(df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]), ['Easting']],
           df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]), ['Northing']],
           #df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]), ['Height']],
           df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]), ['Height']],
           pdf_list,
           c='yellow')

plt.show()


# In[126]:


min(pdf_list), max(pdf_list)


# In[99]:


df_train_conv.loc[df_train_conv['Class'].isin([class_dict['rails']]), ['Easting', 'Northing', 'Height']]


# In[ ]:




