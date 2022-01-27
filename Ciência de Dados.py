# ### Importar Bibliotecas e Bases de Dados

# In[1]:


import pandas as pd
import pathlib 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import time


# In[2]:


base_airbnb = pd.DataFrame()

caminhos_base = pathlib.Path('dataset')

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

for arquivo in caminhos_base.iterdir():
    
    
    nome_mes = arquivo.name[:3]
    mes = meses[f'{nome_mes}']
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    
    print(arquivo)
    df = pd.read_csv(caminhos_base / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
display(base_airbnb)


# - Como temos muitas colunas nosso modelo pode ficar muito lento
# - Além disso uma analise rapida mostra que muitas colunas não são necessarias para o nosso modelo de previsão,por isso, vamos excluir os algumas colunas na base
# - Tipo de colunas que vamos excluir:
#     1. ID's, links e informações não relevantes para o modelo
#     2. Colunas repetidas ou extremamentes parecidas com outras que passam a mesma informação ao modelo, Ex: data vs ano/mes
#     3. Colunas preenchidas com texto livre, não rodaremos nenhuma analise de palavras ou algo do tipo
#     4. Colunas onde todos ou quase todos os valores são iguais
#     
#     
# - para isso vamos criar um arquivo em excel com os mil primeiros registros e fazer uma analise qualitativa

# In[3]:


base_airbnb.head(1000).to_csv('Primeiros_registros.csv')


# ### Consolidar Base de Dados

# In[4]:


colunas = ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes']
print(colunas)


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir
# - Depois da analise qualitativa das colunas, levando em conta os critérios ja citados acima, ficamos com as seguintes colunas

# In[5]:


base_airbnb = base_airbnb.loc[:, colunas]
display(base_airbnb)


# In[6]:


print(base_airbnb.isnull().sum())


# In[ ]:





# ### Tratar Valores Faltando
# - Visualizando os dados, percebemos que existe uma grande dispariedade em dados faltantes. As colunas com mais de 300.000 valores NaN foram excluidas da analise.
# - Para as outras colunas, como temos muitos dados, mais 900.000 linhas, vamos excluir as linhas com NaN

# In[7]:


for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())


# ### Verificar Tipos de Dados em cada coluna

# In[8]:


print(base_airbnb.info())
print('--' * 30)
print(base_airbnb.iloc[0])
#price


# COMO PRICE E EXTRA PEOPLE ESTÃO SENDO LIDAS COMO OBJECT VAMOS TRANFORMA-LAS EM FLOAT

# In[9]:


#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float64, copy=False)

#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float64, copy=False)

print(base_airbnb.info())


# ### Análise Exploratória e Tratar Outliers

# - Basicamente olha feature por feature para:
#     1. Ver a correlação entre as features e decidir se manteremos todas as features que temos
#     2. Excluir outliers(usaremos como regra, valores abaixo de Q1 - 1.5 x Amplitude e valores acima de Q3 + 1,5 x Amplitude). Amplitude = Q3 - Q1
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir 
#     
# - Vamos começar pelas colunas de preço(resultado final que queremos) e de extra_people (também valor monetário). Esse são valores numéricos contínuos
# - Depois vamos avaliar as colunas de valores numéricos discretos(accomodates, bedrooms, guests, included, etc.)
# - Por fim, vamos avaliar as colunas de texto e definir quais categoria são importantes e se devemos mantes ou não.
# 
# CUIDADO: não saia excluindo direto outliers, pense exatamente no que você está fazendo. Se não tem um motivo claro para remover o outlier, talves não seja necessário e pode ser prejudicial para a generalização. Então ter uma balança ai. Claro que você sempre pode testar e ver qual dá o melhor resultado, mas fazer isso para todas as features vai dar muito trabalho
# 
# Ex de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listing_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo e também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do seu modelo. Pense sempre no seu objetivo

# In[10]:


plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot = True, cmap = 'Greens')

#excluir outlies

#print(base_airbnb.corr())


# ## Definição de funcões para Análise de outliers
# 
# vamos definir algumas função para ajudar na analise outliers das colunas

# In[11]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - (1.5 * amplitude), q3 + (1.5 * amplitude)
def excluir_outliers(df, coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[coluna])
    df = df.loc[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


# In[12]:


def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)
    
def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# ### Coluna de Price

# In[13]:


diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])


# Como estamos construindo um modelo para imoveis comuns, acredito que o valores acima do limite superior são de aprtamentos de altissimo luxo, que não é nosso objetivo principal, por isso podemos deletar esses outliers

# In[14]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'price')
print(f'{qtde_linhas} removidas na coluna de price')
histograma(base_airbnb['price'])


# ### Coluna Extra People

# In[15]:


diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])


# In[16]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'extra_people')
print(f'{qtde_linhas} removidas na coluna de extra_people')
histograma(base_airbnb['extra_people'])


# ### host_listings_count 

# In[17]:


diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])


# In[18]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'host_listings_count')
print(f'{qtde_linhas} removidas na coluna de host_listings_count')


# Podemos excluir os outliers, porque host com mais de 6 imoveis no airbnb não é o publico do modelo (imagino que estes sejam imobiliarias ou profissionais que gerenciam imoveis no airbnb)

# ### accommodates

# In[19]:


diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])


# In[20]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'accommodates')
print(f'{qtde_linhas} removidas na coluna de accommodates')


# ### bathrooms

# In[21]:


diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())


# In[22]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'bathrooms')
print(f'{qtde_linhas} removidas na coluna de bathrooms')


# In[ ]:





# ### bedrooms

# In[23]:


diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])


# In[24]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'bedrooms')
print(f'{qtde_linhas} removidas na coluna de bedrooms')


# ### beds

# In[25]:


diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])


# In[26]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'beds')
print(f'{qtde_linhas} removidas na coluna de beds')


# ### guests_included

# In[27]:


#diagrama_caixa(base_airbnb['guests_included'])
#grafico_barra(base_airbnb['guests_included'])
print(limites(base_airbnb['guests_included']))
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())


# Vamos remover essa featura da analise, parece que os usuarios usam muito o valor padrão do airbnb sendo 1 guest_included. Isso pode levar o nosso modelo a considerar uma feature que não afetara a definição de preço

# In[28]:


base_airbnb = base_airbnb.drop('guests_included', axis=1)
print(base_airbnb.shape)


# ### minimum_nights

# In[29]:


diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])


# In[30]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'minimum_nights')
print(f'{qtde_linhas} removidas na coluna de minimum_nights')


# ### maximum_nights

# Como o numero maximo de noites tem como usa maioria o padrão 0 do airbnb e alguns valores soltos por volta de dois mil, faz com que a feature seja inutilizavel

# In[31]:


base_airbnb = base_airbnb.drop('maximum_nights', axis = 1)
print(base_airbnb.shape)


# ### number_of_reviews

# In[32]:


diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])


# Como o modelo é constuido para uma pessoa que esta a entrar no airbnb, é esperado que ele não tenha review

# In[33]:


base_airbnb = base_airbnb.drop('number_of_reviews', axis = 1)
print(base_airbnb.shape)


# ## Tratamento de colunas de Texto

# ### property_type 

# In[34]:


plt.figure(figsize = (15, 5))

grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# In[35]:


tabela_tipo_casa =  base_airbnb['property_type'].value_counts()
colunas_agrupar = []
for tipo in tabela_tipo_casa.index:
    if tabela_tipo_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type'] = 'Outros'

print(base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# In[ ]:





# ### room_type 

# In[36]:


plt.figure(figsize = (15, 5))

grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# In[ ]:





# ###  bed_type

# In[37]:


print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize = (15, 5))

grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# In[38]:


tabela_camas = base_airbnb['bed_type'].value_counts()
tabela_juntas = []
for tipo in tabela_camas.index:
    if tabela_camas[tipo] < 10000:
        base_airbnb.loc[base_airbnb['bed_type'] == tipo, 'bed_type'] = 'Outras'

print(base_airbnb['bed_type'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# ### cancellation_policy 

# In[39]:


plt.figure(figsize = (15, 5))

grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# In[40]:


print(base_airbnb['cancellation_policy'].value_counts())


# In[41]:


tabela_politica = base_airbnb['cancellation_policy'].value_counts()
tabela_juntas = []
for tipo in tabela_politica.index:
    if tipo in ['strict', 'super_strict_60', 'super_strict_30']:
        base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict'

print(base_airbnb['cancellation_policy'].value_counts())

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation= 90)


# ### amenities 

# In[42]:


base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb.drop('amenities', axis=1)
display(base_airbnb)


# In[43]:


diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])


# In[44]:


base_airbnb, qtde_linhas = excluir_outliers(base_airbnb, 'n_amenities')
print(f'{qtde_linhas} removidas na coluna de n_amenities')
grafico_barra(base_airbnb['n_amenities'])

base_airbnb = base_airbnb.drop('amenities', axis = 1)


# # Visualização de Mapa das Prioridades

# In[45]:


amostra = base_airbnb.sample(n=50000)


centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}

mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price',zoom=10,
                         center=centro_mapa, mapbox_style='stamen-terrain', radius=2.5)

mapa.show()


# In[ ]:





#         
# 

# ### Encoding

# Precisamos aJustar as features para facilitar o trabalho do modelo futuro (features de categoria, True or False, etc)
# ##### Em features com true e false, vamos substituir true por 1 e false por 0
# ##### Features de Categoria (em que os valores da coluna são textos) vamos utilizar o método de enconding váriaveis dummies

# In[46]:


colunas_df = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_df:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0
    
print(base_airbnb_cod.iloc[0])


# In[47]:


coluna_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns= coluna_categorias)
display(base_airbnb_cod.head())


# ### Modelo de Previsão

# In[48]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}\nR²:{r2}\nRSME:{RSME}\n'


# - Escolha dos modelos a serem testados
# 
# 1. Random Forest
# 2. LinearRegression
# 3. ExtraTree

# In[49]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
           'LinearRegression': modelo_lr,
           'ExtraTrees': modelo_et,
          }

X= base_airbnb_cod.drop('price', axis=1)
y= base_airbnb_cod['price']


# - Separar Dados de Treino e Teste + Treino do modelo

# ### Análise do Melhor Modelo

# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    #treinar modelo
    modelo.fit(X_train, y_train)
    #teste modelo
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - Modelo escolhido como o melhor modelo foi o ExtraTreesRegressor
# 
#     Esse foi o modelo com maior valor de R² e ao mesmo tempo com o menor valor de RSME. Como não tivemos uma grande diferença de velocidade de tempo e previsão desse modelo com o modelo de RandomForest, que teve valor de R² e RSME)
#     
#     Como o modelo de de LinearRegression não teve um resultado satisfatório, com valores de R² e RSME muito piores do que os outros modelos

# In[ ]:





# In[51]:


importancia_features = pd.DataFrame(data=modelo_et.feature_importances_, index=X_test.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)
display(importancia_features)

plt.figure(figsize=(15,5))
grafico = sns.barplot(x=importancia_features.index, y=importancia_features[0])
grafico.tick_params(axis='x', rotation= 90)


# ### Ajustes e Melhorias no Melhor Modelo
# 
# - is_business_travel_ready não parece ter muito impacto no nosso programa, por isso para chegar à um modelo mais simples vamos excluir essa feature
# 
# - resultado original:<br><br> R²:0.9751366496806467,<br>RSME:41.81325608182725<br><br>
# 

# In[52]:


base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis = 1)
base_teste = base_airbnb_cod.copy()
for c in base_teste:
    if 'bed_type' in c:
        print(c)
        base_teste = base_teste.drop(c, axis=1)


# In[53]:


X= base_teste.drop('price', axis=1)
y= base_teste['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('extra_trees', y_test, previsao))


# In[54]:


print(base_teste.columns)


# ## Deploy do Projeto
# 
# - Passo 1: -> Criar Arquivo em Modelo(joblib)
# - Passo 2: -> Escolher a Forma de Deploy:
#     - Arquivel Executável + Tkinter
#     - Deploy em um Microsite (Flask)
#     - Deploy apenas para Uso Direto(Streamlit)
# - Passo 3: -> Outro Arquivo Python (pode ser Jupyter ou Pycharm)
# - Passo 4: -> Importar o Streamlit e criar código
# - passo 5: -> Atribuir ao botão o carregamento do modelo
# - passo 6: -> Deploy Feito!

# In[55]:


X['price'] = y
X.to_csv('dados.csv')


# In[56]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')

