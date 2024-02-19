#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("school-shootings-data.csv")
pd.set_option('display.max_columns', None)
print(df.info())
print(df.head())
print(df.isnull().sum())



# In[2]:


df = df.dropna()
print(df.isnull().sum())


# In[3]:


import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv("school-shootings-data.csv")
pd.set_option('display.max_columns', None)
spc = df.groupby('school_name')['killed'].sum().reset_index() # группировка
ts = spc.sort_values(by='killed', ascending=False).head(15)  # сортировка
fig = go.Figure()  
fig.add_trace( 
    go.Bar(x=ts['killed'], y=ts['school_name'],marker=dict(color=ts['killed'], coloraxis="coloraxis"),marker_line_width=2, marker_line_color="black", orientation='h',) # оси х,у/цвет столбцов/толщина границ/цвет границ
)
fig.update_layout(  # Настройка параметров диаграммы и меток
    title=dict( text="Топ-10 годов по количеству убийств", x=0.5,y=0.95, font=dict(size=20)), # заголовок/ положение текста/ размер текста
    xaxis=dict(title="Количество убийства",  title_font=dict(size=16),tickangle=315,  showgrid=True,  gridwidth=2,  gridcolor='ivory',  tickfont=dict(size=14)), # ось х/ размер текста/ угол меток/ отобраажение сетки/ширина сетки/цвет
    yaxis=dict(title="Годы",showgrid=False), # ось у/ скрыть
    margin=dict(l=0, r=0, t=50, b=0),  # отступы 
    height=500,  
    coloraxis=dict(colorscale="Inferno"),  # Цветовая шкала
    xaxis_showgrid=True,
)
fig.show()  # Отобразить диаграмму


# In[4]:


import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv("school-shootings-data.csv")
pd.set_option('display.max_columns', None)
kbs = df.groupby('state')['killed'].sum().reset_index()
ts = kbs.sort_values(by='killed', ascending=False).head(10)
fig = go.Figure()  
fig.add_trace(  
    go.Pie(labels=ts['state'],  values=ts['killed'], textinfo='label+percent', textfont_size=14, marker=dict(line=dict(color='black', width=2)))# название стран/убийства/название процент/ граница доли
)
fig.update_layout(
    title=dict(text="Распределение убийств по топ-10 штатам",x=0.5,y=0.95,font=dict(size=20)),
    margin=dict(l=0, r=0, t=50, b=0),
    height=500,
    coloraxis=dict(colorscale="Viridis"),
)
fig.show() 


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("school-shootings-data.csv")
boys = df[df['gender_shooter1'] == 'm']  # Фильтр по полу
girls = df[df['gender_shooter1'] == 'f']
bk = boys.groupby('year')['killed'].sum()  # Сумма убийств для полов
gk = girls.groupby('year')['killed'].sum() 
plt.figure(figsize=(10, 6))  # Устанавливаем размер графика
plt.plot( bk.index, bk.values, marker='o', linestyle='-', color='blue', markersize=8, markerfacecolor='white', markeredgewidth=2, markeredgecolor='black', label='Мальчики') # параметры для мальчиков
plt.plot(gk.index, gk.values,marker='o', linestyle='-', color='crimson', markersize=8,markerfacecolor='white', markeredgewidth=2, markeredgecolor='black', label='Девочки') # параметры для девушек
plt.xlabel('Год')
plt.ylabel('Количество убийств')
plt.title('Зависимость количества убийств от года и пола')
plt.legend()  # Пометка в углу графика
plt.grid(linewidth=2, color='mistyrose')  # Сетка
plt.show()


# In[6]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.cm as cm

mnist = fetch_openml("mnist_784", version=1, parser='auto')# Загружаем MNIST данные
X, y = mnist.data.astype('float32'), mnist.target.astype('int64')# Подготавливаем данные
subset_size, perplexities = 10000, [1, 10, 100]
X_subset, y_subset = X[:subset_size], y[:subset_size]
plt.figure(figsize=(15, 5))# Создаем фигуру для графиков
for i, perplexity in enumerate(perplexities):# Перебираем значения perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X_subset)
plt.subplot(1, len(perplexities), i + 1)# Создаем подграфик
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap=cm.get_cmap("jet", 10), s=10)
plt.title(f"Perplexity = {perplexity}")
plt.xticks([]), plt.yticks([])
plt.colorbar(sc, ticks=range(10), label='Digit')# Добавляем цветовую шкалу
plt.suptitle("MNIST Data Visualization")# Добавляем цветовую шкалу

# Отображаем графики
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import umap
import time
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
mnist = fetch_openml("mnist_784", version=1, parser='auto')# Загрузка данных MNIST
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
subset_size = 10000# Выбор подмножества данных для ускорения вычислений
X_subset = X[:subset_size]
y_subset = y[:subset_size]
n_neighbors_values = [5, 20, 100]# Списки значений параметров n_neighbors и min_dist
min_dist_values = [0.001, 0.01, 0.1]
plt.figure(figsize=(15, 10))# Создание фигуры для графиков
for i, n_neighbors in enumerate(n_neighbors_values):# Создание фигуры для графиков
    for j, min_dist in enumerate(min_dist_values):# Перебор значений min_dist
        umap_start_time = time.time()# Измерение времени выполнения UMA
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)# Измерение времени выполнения UMA
        X_umap = umap_model.fit_transform(X_subset)
        umap_elapsed_time = time.time() - umap_start_time
        tsne_start_time = time.time()# Измерение времени выполнения t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)# Измерение времени выполнения t-SNE
        X_tsne = tsne.fit_transform(X_subset)
        tsne_elapsed_time = time.time() - tsne_start_time
        plt.subplot(len(n_neighbors_values), len(min_dist_values), i * len(min_dist_values) + j + 1)# Создание подграфика
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_subset, cmap=plt.cm.get_cmap("jet", 10), s=10)
        plt.title(f"n_neighbors={n_neighbors}, min_dist={min_dist}\nUMAP time: {umap_elapsed_time:.2f}s, t-SNE time: {tsne_elapsed_time:.2f}s")
        plt.xticks([])
        plt.yticks([])

# Отображение графиков
plt.tight_layout()
plt.show()



