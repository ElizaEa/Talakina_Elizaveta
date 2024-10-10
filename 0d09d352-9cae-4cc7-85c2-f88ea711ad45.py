#!/usr/bin/env python
# coding: utf-8

# <h1> Комментарий ревьювера </h1>
# 
# 👋 **Привет, Лиза!** 
# 
# ✨ Поздравляю с началом работы над финальным проектом!
# 
# Меня зовут Арсен Абдулин и я буду твоим ревьювером. Предлагаю общаться на «ты», если не против =)
# 
# При проверке работ я делаю следующие комментарии:
# 
# <div class="alert alert-success">
# <b>✔️ Зеленым цветом</b> отмечены удачные решения.</div>
# 
# <div class="alert alert-warning">
# <b>⚠️ Желтым цветом</b> я отметил пункты, которые в следующий раз можно сделать по-другому. Одно-два таких замечания в проекте допускается, но если их много — проект следует доработать. </div>
# 
# <div class="alert alert-danger">
# <b>🚫 Красным цветом</b> отмечены критические замечания, которые необходимо поправить, чтобы принять проект. </div>
# 
# Если какие-то моменты в задании для тебя были непонятны и у тебя есть ко мне вопросы — смело спрашивай 😊 Также ты можешь доработать места, где есть желтые комментарии в проекте (однако, это не обязательно).
# 
# Предлагаю работать в диалоге: если ты решишь что-то поменять по моим рекомендациям — пиши об этом (выбери для своих комментариев определенный цвет - так мне будет легче увидеть изменения). Пожалуйста не перемещай, не изменяй и не удаляй мои комментарии. Все это поможет сделать проверку твоего проекта оперативнее.
# 
# <div class="alert alert-info"> <b>ℹ️ Комментарий студента: </b> Пример комментария. </div>

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

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Хорошо, когда есть подробное описание проекта и поставлена цель — так постороннему человеку будет проще ознакомиться!   
# 
# </div>

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


# <div class="alert alert-danger">
# <b>🚫Комментарий ревьювера:</b> Поправь пожалуйста параметр random_state:
#     
#     20924
# 
# </div>

# <div class="alert alert-info"> <b>ℹ️ Комментарий студента: </b> Исправила! Я уже после отправки вспомнила что могла неправильно дату записать.  </div>

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера v2:</b> В целом это не страшно =)
# 
# </div>

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


# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Отлично, получили целевую переменную!
#         
# </div>

# Строки со значением No заменены на ноль, даты прекращения обслуживания заменены на 1.

# Далее сформируем столбец Total Days с общим количеством дней. 

# In[26]:


df_contract['EndDate'] = df_contract['EndDate'].replace(['No'], ['2020-02-01'])


# In[27]:


df_contract['BeginDate'] = pd.to_datetime(df_contract['BeginDate'], format='%Y-%m-%d')
df_contract['EndDate'] = pd.to_datetime(df_contract['EndDate'], format='%Y-%m-%d')


# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Хорошо, так мы получим новый полезный признак для обучения.
# 
# </div>

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

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Хорошо, что обратила внимание на пустые строки в признаке TotalCharges!
#     
# Как рекомендация, пропуски можно заполнить месячными значениями платежей.
#     
# </div>

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


# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Отлично, датасет с данными собран успешно!
#     
# </div>

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

# <div class="alert alert-warning">
# <b>⚠️ Комментарий ревьювера:</b> Как рекомендация, пропуски в услугах интернета можно заполнить значениями 'No'.
#     
# Так мы исключим появление избыточных категорий после кодирования.
#     
# </div>

# <div class="alert alert-info"> <b>ℹ️ Комментарий студента: </b> Ну, мне казалось важным разделить использование и не использование услуг. Исправила.</div>

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера v2:</b> Как вариант, в услугах интернета и телефонии можно заполнить пропуски значением 'no_service', а в остальных услугах просто указать 'No'.
# 
# </div>

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

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Здорово, что применила библиотеку phik для анализа корреляции смешанного набора признаков!
#     
# ***
# 
# Что касается корреляции категориальных признаков, то это пока что пища для размышлений, не стоит сразу по этому условию удалять признаки. Данная корреляция может быть нелинейная, и на линейные модели это не повлияет.   
#         
# </div>

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

# <div class="alert alert-danger">
# <b>🚫 Комментарий ревьювера:</b> В нашем проекте необходимо также сравнить, как отличаются распределения признаков для ушедших и лояльных клиенты.
#     
# Удобно воспользоваться инструментом histplot:
#     
# https://seaborn.pydata.org/generated/seaborn.histplot.html
#     
#     sns.histplot(data, x=column, hue='target', multiple="stack")
#     
# Например, можно посмотреть различия в ежемесячных платежей, а также других признаках
#     
#     
# </div>

# <div class="alert alert-info"> <b>ℹ️ Комментарий студента: </b> Исправила.  </div>

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера v2:</b> Хорошая работа!
#     
# Практически по каждому признаку мы видим отличия в группах ушедших и оставшихся клиентов 👍
# 
# </div>

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


# <div class="alert alert-danger">
# <b>🚫 Комментарий ревьювера:</b> Замечания по пайплайну:
#     
# 1. Убрать заполнение пропусков, т.к. в нашем датасете пропуски уже заполнены. Кроме того, заполнение наиболее частым значением в нашем конкретном случае является неправильным.
#     
# 2. Порядковое кодирование для линейных моделей неправильно использовать, необходимо использовать OHE кодирование. 
#     
# </div>

# <div class="alert alert-info"> <b>ℹ️ Комментарий студента: </b> Исправила.  </div>

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера v2:</b> Хорошо, теперь пайплайн обучения моделей составлен правильно 👌
# 
# </div>

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


# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Здорово, что исследовала важность признаков для лучшей модели 👍
#     
# Так мы сможем увидеть, какие признаки слабо влияют на отток, и в принципе их можно вообще исключить из обучения, и сделать новую модель, отобрав только самые важные признаки. (Пересчитывать с новыми признаками не нужно).
# 
# Анализ важности признаков позволяет не только построить лучшую модель и уменьшить число признаков для обучения. Мы также можем выдать рекомендации бизнесу, изучив влияние признаков на отток, тем самым можно повысить качество определенных услуг, например онлайн-платежи или интернет на оптоволокне.
#     
# ***
#     
# Вот здесь можно немного почитать про интерпретацию важности признаков
#     
# https://webiomed.ru/blog/interpretatsiia-rezultatov-mashinnogo-obucheniia/
#     
# https://habr.com/ru/articles/428213/
#     
# </div>

# In[86]:


display(shap.plots.bar(shap_values, max_display=30))


# <div class="alert alert-warning">
# <b>⚠️ Комментарий ревьювера:</b> Как рекомендация, можно добавить график ROC-AUC на тестовой выборке для лучшей модели.
#     
# </div>

# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Вот небольшие статьи по метрикам в машинном обучении, если заинтересует:
# 
# https://habr.com/ru/company/ods/blog/328372/
#     
# https://habr.com/ru/company/jetinfosystems/blog/420261/
# 
# </div>

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


# <div class="alert alert-success">
# <b>✔️ Комментарий ревьювера:</b> Отличный результат на тестировании, поздравляю!
# 
# </div>

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

# <font color=blue>
# <b> ✔️ 👍Заключение ревьювера:</b> Лиза, было приятно проверять твою работу, она выполнена на хорошем уровне!
# 
# В целом тебе удалось справиться с проектом! Данные из разных таблиц собраны в один датафрейм, удалены аномалии, заполнены пропуски, получены новые признаки, а ненужные признаки удалены. Рассмотрено несколько моделей и сделан перебор параметров. Могу отметить самостоятельность и аналитический подход в выполнении проекта!
#    
# Положительные моменты: 
#     
# - грамотное и понятное оформление кода, аккуратность;
#     
# - аналитический подход — при удалении признаков и формировании новых написано соответствующее обоснование;
#     
# - проведен анализ данных и корреляции признаков;
#     
# - рассмотрены несколько моделей для обучения, выполнен перебор гиперпараметров;
#     
# - удалось достичь высокого значения метрики ROC-AUC.
# 
# Сейчас есть некоторые замечания:
#     
# - поправить пайплайн обучения моделей;
#     
# - добавить исследовательский анализ данных для ушедших и оставшихся клиентов.
#     
# На данном этапе проект практически завершен, я также отметил рекомендации. У нас еще есть время — я отправлю его, чтобы ты могла ознакомиться с проверкой, внести изменения и возможно задать вопросы =)
#     
# Также на будущее стоит иметь ввиду, что метрика ROC-AUC является не единственной, всегда нужно прислушиваться к требованиями бизнеса. Обычно одной метрикой не ограничиваются, смотрят также Precision, Recall, и выбирают те, которые наиболее подходят для конкретных задач. 
#     
# Еще нужно смотреть на важность признаков в модели, чтобы понимать, какие параметры влияют на отток.
#     
# Проверка важности признаков по модели это только один из способов (корреляция в том числе). Используют также аналитические методы проверки гипотез.
#     
# Если есть какие-то вопросы, или нужны пояснения по проекту — смело пиши! Я постараюсь тебе помочь 😊
# 
# Жду твоего ответа!
#     
# </font>

# <div class="alert alert-success">
# <b>✔️ Заключение ревьювера:</b> Лиза, было приятно поработать с тобой над финальным проектом 😊 Видно, что ты серьезно отнеслась к работе, как к реальному проекту!
#     
# В процессе работы над проектом ты проявила себя как грамотный специалист — могу отметить самостоятельность и аналитический подход!
#     
# На мой взгляд работа выполнена на хорошем уровне. По отчету понятно, какие признаки использовались для обучения, как были получены дополнительные признаки, указана модель для обучения и гиперпараметры — по этим шагам можно воспроизвести проект.
#     
# ✨ Поздравляю тебя с успешным завершением финального спринта! Если вспомнить, с чего начиналось обучение, видно как от простых задач мы перешли к реальным проектам, и у тебя это здорово получилось. Я считаю, что это заслуживает похвалы 👍
#     
# Сейчас ты можешь доработать этот проект с учетом рекомендаций, удалить наши с тобой комментарии и добавить себе в портфолио — лучше это сделать в ближайшее время, пока все нюансы еще отложились в памяти =)
#     
# Здесь хочу добавить несколько рекомендаций на будущее:
#     
# - для анализа данных по ситуации можно применять библиотеки pandas_profiling или sweetviz. Когда нужен оперативный отчет по датафрейму, можно воспользоваться такими инструментами для автоматизации. Но в учебных проектах мы делали вручную, поскольку должны были всему научиться;
#     
# - следует отметить, что можно использовать аналитические методы проверки гипотез для оценки влияния признаков на отток. Например, можно воспользоваться инструментом ANOVA для дисперсионного анализа: https://www.reneshbedre.com/blog/anova.html
#     
# ***
# 
# Далее, хочу добавить некоторые ссылки на полезные ресурсы по машинному обучению:
# 
# https://academy.yandex.ru/handbook/ml онлайн-учебник от Школы анализа данных Яндекса, в котором описаны теоретические основы работы моделей машинного обучения;
# 
# https://www.youtube.com/watch?v=xl1fwCza9C8 познавательное видео по настройке модели CatBoost
# 
# https://habr.com/ru/company/ods/blog/322626/ на Habr можно закрепить свои знания, порешав задачи из цикла статей — Открытый курс машинного обучения
# 
# https://github.com/esokolov/ml-course-hse — на гитхаб есть репозиторий с задачами из курса по машинному обучению от Евгения Соколова, можно использовать как дополнительный материал для закрепления знаний.
# 
# https://habr.com/ru/company/avito/blog/571094/ — материалы по A/B тестам.
# 
# Также хочу поделиться опытом и вкратце рассказать о том, в какие направления можно подаваться на работу и какие навыки там пригодятся:
# 
# Направление аналитики: хорошие знания теории вероятностей и математической статистики; базовые знания библиотек ML; уверенные знания SQL; умение решать задачи по A/B тестам; плюсом будет знание специальных инструментов для аналитики по визуализации результатов.
# 
# Направление ML: уверенные знания классических моделей машинного обучения; хорошие знания SQL; понимание алгоритмов машинного обучения — как устроена модель линейной регрессии, модели случайного леса, градиентного бустинга и т.д. На собеседовании могут спросить, как работает какая-нибудь модель «под капотом», очень любят градиентный бустинг и случайный лес. Здесь поможет учебник https://academy.yandex.ru/handbook/ml
# 
# В некоторых компаниях при устройстве на работу или стажировку, например в Яндексе, нужно решить тест на алгоритмические задачи. По алгоритмам есть много разных курсов, платных и бесплатных. Можно попробовать Тренировки по Алгоритмам от Яндекса https://yandex.ru/yaintern/algorithm-training (бесплатно, хороший курс). Насколько знаю, у Практикума тоже есть курс по алгоритмам.
# 
# Есть компании, где требуется умение решать задачи с использованием нейронных сетей, в основном задачи распознавания изображений и анализа текстов (NLP). Если понравилось работать с нейронками, желательно порешать разные задачи, набрать портфолио из нескольких работ, можно подаваться.
# 
# Существуют научные и производственные компании, которые внедряют алгоритмы ML в свои рабочие процессы — в таких компаниях можно совместить должность, например инженера и программиста. Тоже хороший вариант, но нужно понимать, если хотите развиваться дальше, особенно на первых этапах, у вас должны быть опытные наставники, которые помогут советом.
#     
# ***
#     
# Еще раз поздравляю с завершением обучения и желаю успехов в дальнейшем пути Data Scientist'a!
#     
# </div>

# In[ ]:




