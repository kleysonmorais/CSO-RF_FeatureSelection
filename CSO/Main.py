# ------------------------------------------------------------------------------+
#
#   Morais, Kleyson.
#   08, 2018
#
# ------------------------------------------------------------------------------+

from Models import *
from Controller import *
from EvaluationMetric.avaliador import AvaliadorController

# import numpy as np
import pandas as pd
import sys

def inicializa(nome, qtdParticulas):
    # Lendo a Base
    data = pd.read_csv(nome)
    y = data.classe
    list = ['classe']
    X = data.drop(list, axis=1)

    # Ajustando Parâmetros
    avaliarController = AvaliadorController(X, y)
    dadosModel = DadosModel(X, y)

    # Inicializando Enxame
    enxame = EnxameModel()
    enxameController = EnxameController(dadosModel, avaliarController)
    enxameController.criarEnxame(enxame, qtdParticulas)

    return enxame, enxameController, avaliarController

def movimentar(enxame, enxameController, geracoes):
    for i in range(geracoes):
        print("Iteração ", i+1, "/", geracoes)
        enxameController.atualizaEnxame(enxame)


def avaliar(enxame, enxameController, avaliarController):
    enxameController.verificarMelhorPosicaoEnxame(enxame)
    avaliarController.allClassifiers(enxame._melhorPosicaoGlobal)

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Exemplo:")
        print("python Main.py base qtdParticulas geracoes")
        exit()

    nomeBase = "../datasets/"+sys.argv[1]
    qtdParticulas = int(sys.argv[2])
    geracoes = int(sys.argv[3])

    enxame, enxameController, avaliarController = inicializa(nomeBase, qtdParticulas)
    movimentar(enxame, enxameController, geracoes)
    avaliar(enxame, enxameController, avaliarController)

