import numpy as np 
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import time

from Models import *
from Controller import *
from EvaluationMetric.avaliador import AvaliadorController

nome_base_dados = "cancer"

data = pd.read_csv('../data/'+nome_base_dados+'.csv')
list = ['classe']
y = data.classe
X = data.drop(list,axis = 1)
# print(y.value_counts())

tamPopulacao = 10
geracoes = 1000
ac = AvaliadorController(X, y)
# ac.allClassificadores()

dadosModel = DadosModel(X, y)
enxame = EnxameModel()
# ec = EnxameController(dadosModel, nome_base_dados)
ec = EnxameController(dadosModel, avaliador=ac, tipo="wrappers")
ec.criarEnxame(enxame, tamPopulacao)

for i in range(geracoes):
    ec.atualizaEnxame(enxame)

ac.allClassifiers(enxame._melhorPosicaoGlobal)

selecionadas = []
for i, feature in enumerate(enxame._melhorPosicaoGlobal):
    if feature == 1:
        selecionadas.append(i+1)


print('\n',selecionadas)
