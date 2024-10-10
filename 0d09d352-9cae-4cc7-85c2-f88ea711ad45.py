#!/usr/bin/env python
# coding: utf-8


# # Телекоммуникации «ТелеДом»

# _____
# **Описание исследования.**
# 
# Оператор связи «ТелеДом» хочет бороться с оттоком клиентов. Для этого его сотрудники начнут предлагать промокоды и специальные условия всем, кто планирует отказаться от услуг связи. Чтобы заранее находить таких пользователей, «ТелеДому» нужна модель, которая будет предсказывать, разорвёт ли абонент договор. Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и услугах.
# 
# Оператор предоставляет два основных типа услуг: 
# 
# 
# - Стационарную телефонную связь. Телефон можно подключить к нескольким линиям одновременно.
# 
# - Интернет. Подключение может быть двух типов: через телефонную линию (DSL, от англ. digital subscriber line — «цифровая абонентская линия») или оптоволоконный кабель (Fiber optic).
# 
# 
# Также доступны такие услуги:
# 
# 
# - Интернет-безопасность: антивирус (DeviceProtection) и блокировка небезопасных сайтов (OnlineSecurity).
# 
# - Выделенная линия технической поддержки (TechSupport).
# 
# - Облачное хранилище файлов для резервного копирования данных (OnlineBackup).
# 
# - Стриминговое телевидение (StreamingTV) и каталог фильмов (StreamingMovies).
# 
# 
# Клиенты могут платить за услуги каждый месяц или заключить договор на 1–2 года. Возможно оплатить счёт разными способами, а также получить электронный чек.
# 
# 
# _____
# **Цели исследования.**
#     
# Построить модель для прогноза оттока клиентов.
# 
# 
# _____
# **Исходные данные.**
# 
# 
# Данные состоят из нескольких файлов, полученных из разных источников:
#  
# 1. **contract_new.csv** — информация о договоре.
# 
# 
# 2. **personal_new.csv** — персональные данные клиента.
# 
#   
# 3. **internet_new.csv** — информация об интернет-услугах.
# 
# 
# 4. **phone_new.csv** — информация об услугах телефонии.
#    
# 
# 
# Файл contract_new.csv:
# 
# 
# 
# - **customerID** — идентификатор абонента.
# 
# 
# - **BeginDate** — дата начала действия договора.
# 
# 
# - **EndDate** — дата окончания действия договора.
# 
# 
# - **Type** — тип оплаты: раз в год-два или ежемесячно.
# 
# 
# - **PaperlessBilling** — электронный расчётный лист.
# 
# 
# - **PaymentMethod** — тип платежа.
# 
# 
# - **MonthlyCharges** — расходы за месяц.
# 
# 
# - **TotalCharges** — общие расходы абонента.
# 
# 
# 
# Файл personal_new.csv:
# 
# 
# 
# - **customerID** — идентификатор пользователя.
# 
# 
# - **gender** — пол.
# 
# 
# - **SeniorCitizen** — является ли абонент пенсионером.
# 
# 
# - **Partner** — есть ли у абонента супруг или супруга.
# 
# 
# - **Dependents** — есть ли у абонента дети.
# 
# 
# 
# Файл internet_new.csv:
# 
# 
# 
# - **customerID** — идентификатор пользователя.
# 
# 
# - **InternetService** — тип подключения.
# 
# 
# - **OnlineSecurity** — блокировка опасных сайтов.
# 
# 
# - **OnlineBackup** — облачное хранилище файлов для резервного копирования данных.
# 
# 
# - **DeviceProtection** — антивирус.
# 
# 
# - **TechSupport** — выделенная линия технической поддержки.
# 
# 
# - **StreamingTV** — стриминговое телевидение.
# 
# 
# - **StreamingMovies** — каталог фильмов.
# 
# 
# 
# Файл phone_new.csv:
# 
# 
# 
# - **customerID** — идентификатор пользователя.
# 
# 
# - **MultipleLines** — подключение телефона к нескольким линиям одновременно.
# 
# 
# 
# _____
# **Данное исследование разделим на четыре этапа:**
# 
# 
# 
# 1. [Выполнить загрузку, исследовательский анализ и предобработку данных.](#section2) 
# <a href='#section2'></a>
# 
# 
# 2. [Объединить данные.](#section3)
# <a href='#section3'></a>
# 
# 
# 3. [Провести исследовательских анализ объединенных данных.](#section4)
# <a href='#section4'></a>
# 
# 
# 5. [Обучить модели](#section5), выбрать и [протестировать лучшую модель.](#section6) 
# <a href='#section5'></a>
# <a href='#section6'></a>
# 
# 



# ### Библиотеки и окружение

# In[1]:


get_ipython().system('pip install numba -U -q')


# In[2]:


get_ipython().system('pip install numpy==1.24.4 -q')


# In[3]:


get_ipython().system('pip install matplotlib==3.8.4 -q')


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().system('pip install shap==0.45.1 -q')


# In[6]:


get_ipython().system('pip install scikit-learn==1.4.0 -q')


# In[7]:


pip install phik==0.12.4 -q


# In[8]:


get_ipython().system('pip install seaborn==0.13.2 -q')


# In[ ]:


get_ipython().system('pip install plot-metric -q')


# In[9]:


import seaborn as sns


# In[10]:


import numpy as np
import pandas as pd
import math


# In[11]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt


# In[13]:


import phik
from phik import phik_matrix
from phik.report import plot_correlation_matrix


# In[14]:


import shap


# In[95]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[16]:


from sklearn.pipeline import Pipeline


# In[104]:


from plot_metric.functions import BinaryClassification


# **Функции:**

# In[17]:


def check(row):
    col = row.columns
    dupli = row.duplicated().sum()
    print('Количество явных дубликатов:', dupli)
    nans = row.isna().sum().sum()    
    print('Количество пропусков в данных:', nans)
    if dupli > 0:
        print(row.duplicated())
    for col_l in col:
        print('-'* 30)
        print('Уникальные значения:', col_l, row[col_l].sort_values().unique())


# In[18]:


# Вывод гистограмм, pie-диаграмм, сводных таблиц
def sign(row,column, title):
    if row.dtypes[column] == object:
        row[column].value_counts().plot.barh(figsize=(10, 5), grid= True)
        plt.title(title, fontsize=15)
        plt.ylabel('Количество')
        plt.rc('xtick', labelsize= 15 ) 
        plt.rc('ytick', labelsize= 15 ) 
        plt.show()
        one = pd.pivot_table(
            row,
            index=column,
            values = 'customerID',
            aggfunc='count')
        plt.title(title, fontsize=20)
        row[column].value_counts().plot(
            kind = 'pie',
            figsize=(10,10),
            autopct = '%0.01f%%',
            colors=sns.color_palette('Set2')
        )
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()
        one['% '] = round(one['customerID']/one['customerID'].sum()*100,2)
        one.rename(columns={'customerID':'количество'},
              inplace=True)
        print(one)
        #return one

    else:
        fig, axes = plt.subplots(2, figsize=(15, 15), sharey='row')
        #axes[0].plot(row[column], kind='bar')
        axes[0].hist(row[column], bins=int(len(row) ** (0.5)))
        axes[0].grid(True)
        axes[0].set_title(title, fontsize=20)
        axes[0].set_ylabel('Количество')
        axes[1].boxplot(data=row, x=column, vert=False, patch_artist=True)
        axes[1].grid(True)
        axes[1].set_xlabel(title)
        plt.show()
        one = row[column].describe().T
        print(one)
        #return one


# In[19]:


def pie(row, column ):
    row[column].value_counts().plot.barh(figsize=(10, 5), grid= True)
    plt.title(column, fontsize=15)
    plt.ylabel('Количество')
    plt.rc('xtick', labelsize= 15 ) 
    plt.rc('ytick', labelsize= 15 ) 
    plt.show()
    one = pd.pivot_table(
        row,
        index=column,
        values = 'customerID',
        aggfunc='count')
    plt.title(column, fontsize=20)
    row[column].value_counts().plot(
        kind = 'pie',
        figsize=(10,10),
        autopct = '%0.01f%%',
        colors=sns.color_palette('Set2')
    )
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
    one['% '] = round(one['customerID']/one['customerID'].sum()*100,2)
    one.rename(columns={'customerID':'количество'},
               inplace=True)
    return one


# In[20]:


def phikmatrix(row, col):
    if col in row.columns:
        phik_overview = (row.drop(col, axis=1).phik_matrix(verbose=False))
    else:
        phik_overview = (row.phik_matrix(verbose=False))
    matrix = np.triu(phik_overview .corr())
    plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    vmin=0, vmax=1, color_map='RdPu',
    title=r'Корреляция $\phi_K$',
    fontsize_factor=1.5,
    figsize=(20, 15)
) 


# In[21]:


RANDOM_STATE = 20924
TEST_SIZE = 0.25



# <a id='section2'></a>
# ## Загрузка, исследовательский анализ и предобработка данных.

# ### df_contract

# In[22]:


df_contract = pd.read_csv('/datasets/contract_new.csv')


# In[23]:


df_contract.info()


# In[24]:


df_contract.head()


# Создадим целевой признак Target.

# In[25]:


df_contract['Target'] = (df_contract['EndDate'] != 'No').astype(int)



# Строки со значением No заменены на ноль, даты прекращения обслуживания заменены на 1.

# Далее сформируем столбец Total Days с общим количеством дней. 

# In[26]:


df_contract['EndDate'] = df_contract['EndDate'].replace(['No'], ['2020-02-01'])


# In[27]:


df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], format='%Y-%m-%d')
df_contract['EndDate'] = pd.to_datetime(df_contract['EndDate'], format='%Y-%m-%d')



# In[28]:


df_contract['TotalDays'] = (df_contract['EndDate'] - df_contract['BeginDate']).dt.days


# In[29]:


df_contract.info()


# In[30]:


df_contract.head()


# In[31]:


check(df_contract)


# In[32]:


drop = df_contract[df_contract['TotalCharges'].str.contains(" ")]
drop


# In[33]:


len(drop)


# 11 строк с пробелом в значении TotalCharges. Все пользователи новые, первая оплата ещё не состоялась.
# 
# Заменим на 0.


# In[34]:


df_contract['TotalCharges'] = df_contract['TotalCharges'].replace({' ':0})


# In[35]:


df_contract['TotalCharges'] = df_contract['TotalCharges'].astype('float64')


# In[36]:


df_contract.info()


# In[37]:


print('Уникальные значения:', df_contract['TotalCharges'].sort_values().unique())


# In[38]:


columns = ['Type', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'TotalDays']


# In[39]:


for column in columns:
    sign(df_contract,column, column)
    


# ##### Вывод:

# - В основном пользователи оформляют месячную подписку(55%)
# 
# - Пользователи чаще пердпочитают оплачивать электронным чеком. Остальные варианты оплаты распределены в равных долях.
# 
# - В среднем пользователи оплачивают услуги в размере 70 у.е. в месяц.

# ###  df_personal

# In[40]:


df_personal = pd.read_csv('/datasets/personal_new.csv')


# In[41]:


df_personal.info()


# In[42]:


df_personal.columns


# In[43]:


df_personal.head()


# In[44]:


columns = ['gender', 'Partner', 'Dependents']


# In[45]:


for column in columns:
    sign(df_personal,column, column)


# ##### Вывод:

# - Услуги в равной мере используют женщины(49.52%) и мужчины(50.48%)
# 
# 
# - Чаще услуги используют люди без детей(70.04%)

# ###  df_internet

# In[46]:


df_internet = pd.read_csv('/datasets/internet_new.csv')


# In[47]:


df_internet.info()


# In[48]:


df_internet.head()


# In[49]:


df_internet.columns


# In[50]:


columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


# In[51]:


for column in columns:
    sign(df_internet,column, column)
    


# ##### Вывод:

# 
# - Тип подключения сравнительно чаще Fiber Optics(56.12%).
# 
# 
# - Блокировку опасных сайтов в 63.4% случаев не подключают.
# 
# 
# - Облачное хранилище подключают в 44.03% случаев.
# 
# 
# - Антивирус от компании используют 43.9%.
# 
# 
# - Техническую поддержку использует 37.05% клиентов.
# 
# 
# - Стриминговое телевидение и каталог фильмов сравнительно одинаково используют или не используют.

# ###  df_phone

# In[52]:


df_phone = pd.read_csv('/datasets/phone_new.csv')


# In[53]:


df_phone.info()


# In[54]:


df_phone.head()


# In[55]:


sign(df_phone,'MultipleLines', 'MultipleLines')
    


# ##### Вывод:

# - Подключение телефона к нескольким линиям одновременно сравнительно чаще не подключают(53.29%)

# #### Вывод:
# Преобразованы:
# 
# - Столбцы 'BeginDate'и'EndDate' приведены к формату даты.
# 
# - TotalCharges приведён к типу float.
# 
# 
# В столбце TotalCharges 11 строк с пробелом в значении заменены на 0.
# 
# Создан столбец Target, содержащий целевую переменную в формате int.
# 
# Создан столбец 'TotalDays', содержащий количество дней использования услуг.

# <a id='section3'></a>
# ## Объединение данных

# In[56]:


df_full = pd.merge(df_contract, df_personal, on='customerID', how='outer')
df_full_internet = pd.merge(df_full, df_internet, on='customerID', how='outer')


# In[57]:


df_full_internet.head()


# In[58]:


df_full_phone = pd.merge(df_full, df_phone, on='customerID', how='outer')


# In[59]:


df_full_phone.head()


# In[60]:


df_full_phone_int = pd.merge(df_full_internet, df_phone, on='customerID', how='outer')


# In[61]:


df_full_phone_int.head()



# Пропуски возникли потому что пользователи не испольют услуги телефонии. 
# 
# Заменим пропуск на значение No.

# In[62]:


df_full_phone_int['MultipleLines'] = df_full_phone_int['MultipleLines'].fillna('No')


# In[63]:


check(df_full_phone_int)


# Пропуски возникли потому что пользователи не подключали интернет. 
# 
# Заменим пропуск на значение No


# In[64]:


df_full_phone_int = df_full_phone_int.fillna('No')


# In[65]:


check(df_full_phone_int)


# #### Вывод:
# 
# - Данные объединены успешно, пропуски заполнены соответствующими значениями.

# <a id='section4'></a>
# ## Исследовательских анализ объединенных данных

# In[66]:


pie(df_full_phone_int,'Target')


# Отток клиентов на данный момент составляет 15.63%

# In[67]:


phikmatrix(df_full_phone_int, 'customerID')


# Оценка корреляции по шкале Чеддока, где:
# 
# - Очень слабая связь — от 0,1 до 0,3.
# 
# - Слабая связь — от 0,3 до 0,5.
# 
# - Средняя связь — от 0,5 до 0,7.
# 
# - Высокая связь — от 0,7 до 0,9.
# 
# - Очень высокая связь — от 0,9 до 1,0.
# 
# **При оценке корреляции с целевым признаком oчень высокая связь**:
# 
# - EndDate
# 
# 
# Мультиколлинеарные пересекающиеся признаки:
# - InternetService 
# - OnlineSecurity  
# - OnlineBackup    
# - DeviceProtection 
# - TechSupport  
# - StreamingTV 
# - StreamingMovies   
# 
# Пары мультиколлинеарных признаков:
# - BeginDate - TotalDays


# In[68]:


df_full_phone_int.set_index("customerID", inplace = True)


# In[69]:


del df_full_phone_int['EndDate']
del df_full_phone_int['BeginDate']
#del df_full_phone_int['TotalCharges']


# In[70]:


df_full_phone_int.head()


# In[71]:


df_full_phone_int.info()


# In[72]:


def hiplot(row):
    columns = row.columns
    for column in columns:
        if column != 'Target':
            plt.figure(figsize=(15, 15))
            sns.histplot(df_full_phone_int, x=column, hue='Target', multiple="stack")
            plt.title(f'Распределения признаков для ушедших и лояльных клиентов\n {column}', fontsize=15)
            plt.ylabel('Количество')
            plt.show()


# In[73]:


hiplot(df_full_phone_int)


# На основе полученых данных сформулируем портрет пользователя, склонного прекратить использование услуг:
# 
# - Человек, оформивший месячную подписку или контракт на 2 года, получающий электронный расчётный чек. Является пенсионером, у клиента есть партнёр и дети. Тип подключения интернета- Fiber Optic, не подключавший блокировку опасных сайтов и выделенную линию технической поддержки. Использует стриминговое телевидение или каталог фильмов. Использует подключение телефона к нескольким линиям одновременно.


# #### Вывод:
# 

# Отток клиентов на данный момент составляет 15.63%
# 
# На основе полученых данных сформулирован портрет пользователя, склонного прекратить использование услуг:
# 
# - Человек, оформивший месячную подписку или контракт на 2 года, получающий электронный расчётный чек. Является пенсионером, у клиента есть партнёр и дети. Тип подключения интернета- Fiber Optic, не подключавший блокировку опасных сайтов и выделенную линию технической поддержки. Использует стриминговое телевидение или каталог фильмов. Использует подключение телефона к нескольким линиям одновременно.

# ## Подготовка данных

# In[75]:


num_col = []
cat_col = []
def types(row, col, id):
    columns = row.columns
    for column in columns:
        if column != col and column != id:
            if row.dtypes[column] == object:
                cat_col.append(column)
            else:
                num_col.append(column)
    return num_col, cat_col


# In[76]:


types(df_full_phone_int, 'Target', 'customerID')


# In[77]:


X = df_full_phone_int.drop(['Target'], axis=1)
y = df_full_phone_int['Target']

X, X_test, y, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# In[78]:


print(f'Размер тестовой выборки: ', f'{(len(X_test)/len(df_full_phone_int)):.0%}')
print(f'Размер тренировочной выборки: ', f'{(len(X)/len(df_full_phone_int)):.0%}')


# In[88]:


param_grid = [
    {   'models': [SVC()],
        'models__kernel': ['rbf', 'sigmoid'],
        'models__C': range(1,5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
        
    },
    {
        'models': [KNeighborsClassifier()],
        'models__n_neighbors': range(2, 5),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {
        'models': [LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', penalty='l1')],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },

    {
        'models': [CatBoostClassifier(verbose=0, learning_rate=0.5)],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
    }
    
]



# In[89]:


ohe_pipe = Pipeline(
    [
        (
            'ohe', 
            OneHotEncoder(sparse_output=False,drop='first')
        )
    ]
)

# создаём общий пайплайн для подготовки данных
data_preprocessor = ColumnTransformer(
    [('ohe', ohe_pipe, cat_col),
     ('num', MinMaxScaler(), num_col)
    ],  
    remainder='passthrough'
)

pipe_final = Pipeline([
    ('preprocessor', data_preprocessor),
    ('models',  DecisionTreeClassifier(random_state=RANDOM_STATE))
     ]
    )


# In[90]:


gscv = GridSearchCV(
    pipe_final,
    param_grid,
    scoring='roc_auc',
    n_jobs=-1,
    cv=3,
    verbose=1,
    error_score='raise'
)


# <a id='section5'></a>
# ## Обучение моделей

# In[91]:


gscv.fit(X, y)


# In[92]:


print('Лучшая модель и её параметры:\n\n', gscv.best_estimator_)
print ('Метрика ROC-AUC лучшей модели на кросс-валидации:', gscv.best_score_)

y_test_pred = gscv.best_estimator_.predict(X_test)


# In[84]:


model = gscv.best_estimator_.named_steps['models']


# In[85]:


X_train_new = pipe_final.named_steps['preprocessor'].fit_transform(X)
explainer = shap.TreeExplainer(model, X_train_new)

X_test_new = pipe_final.named_steps['preprocessor'].transform(X_test)
feature_names = pipe_final.named_steps['preprocessor'].get_feature_names_out()

X_test_new = pd.DataFrame(X_test_new, columns=feature_names)
shap_values = explainer(X_test_new)

display(shap.plots.beeswarm(shap_values, max_display=30))


# In[86]:


display(shap.plots.bar(shap_values, max_display=30))



# <a id='section6'></a>
# ## Выбор и тестирование лучшей модели

# In[107]:


pred_prob = gscv.best_estimator_.predict_proba(X_test)[:, 1]
bc = BinaryClassification(y_test, pred_prob, labels=["Class 1", "Class 2"])

plt.figure(figsize=(15,15))
bc.plot_roc_curve()
plt.show()


# Модель имеет высокую площадь под кривой, равную 0.89, что указывает на то, что она имеет большую площадь под кривой и является лучшей моделью для правильной классификации наблюдений по категориям.

# Проведём расчёт показателя метрики модели на тренировочной выборке

# In[93]:


print ('Метрика ROC-AUC лучшей модели на кросс-валидации:', gscv.best_score_)
print(f'Метрика ROC-AUC на тестовой выборке: {sqrt(roc_auc_score(y_test, y_test_pred))}')



# **Выводы о значимости признаков:**
# 
# 
# **Мало значимые для модели признаки:**
# 
# - TotalCharges
# 
# - MountlyCharges
# 
# - TotalDays
# 
# **Признаки сильнее всего влияющие на целевой признак топ 4:**
# 
# 
# - StreamingMovies — каталог фильмов.
# 
# 
# - Type — тип оплаты.
# 
# 
# - Dependentens — есть ли у абонента дети.
# 
# 
# - Partner — есть ли у абонента супруг или супруга.
# 
# 
# 
# 
# 
# 
# 

# ## Общий вывод

# Данные прошли проверку и соответствуют описанию, явных дубликатов не обнаружено.
#         
#         
# Обнаружены пропуски в данных:
# 
# - В столбце TotalCharges 11 строк с пробелом в значении заменены на 0.
# 
# Для поиска лучшей модели произведён исследовательский и корреляционный анализ данный, далее задействован пайплайн для подбора лучшей модели.
# 
#    **В ходе исследования обнаружено:**
#    - В среднем пользователи оплачивают услуги в размере 70 у.е. в месяц.
#    
#    
#    - Чаще услуги используют люди без детей(70.04%)
#    
#    
#    - Техническую поддержку использует 37.05% клиентов.
#    
#    
#    - Отток клиентов на момент выгрузки датасета составляет 15.63%.
#    
#    
#    
# **Выделена лучшая модель: CatBoostClassifier**
# 
# Задействованная метрика оценки модели -  площадь под ROC-кривой.
# 
# 
# Получены результаты:
# 
# 
# Метрика ROC-AUC на кросс-валидации:  0.8898440489870613
# 
# 
# Метрика ROC-AUC на тестовой выборке: 0.8615676312092809
# 
# 
# 
# 
# 
# Наиболее важные для модели метрики:
# 
# - StreamingMovies — каталог фильмов.
# 
# 
# - Type — тип оплаты.
# 
# 
# - Dependentens — есть ли у абонента дети.
# 
# 
# - Partner — есть ли у абонента супруг или супруга.
# 
# 
# 
# 
# На основе полученых данных сформулирован портрет пользователя, склонного прекратить использование услуг:
# 
# - Человек, оформивший месячную подписку или контракт на 2 года, получающий электронный расчётный чек. Является пенсионером, у клиента есть партнёр и дети. Тип подключения интернета- Fiber Optic, не подключавший блокировку опасных сайтов и выделенную линию технической поддержки. Использует стриминговое телевидение или каталог фильмов. Использует подключение телефона к нескольким линиям одновременно.


