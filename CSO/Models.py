#------------------------------------------------------------------------------+
#
#   Morais, Kleyson.
#   08, 2018
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

# Representação da Partícula
class ParticulaModel:
    _posicao                 = None
    _velocidade              = None
    _fitness                 = None

    def __init__(self):
        self._posicao               = None
        self._velocidade            = None
        self._fitness               = None

# Representação do Enxame
class EnxameModel:
    _particulas                     = None
    _melhorPosicaoGlobal            = None
    _melhorFitness                  = None

    def __init__(self):
        self._particulas                  = []
        self._melhorPosicaoGlobal         = None
        self._melhorFitness               = None

# Representação dos Dados
class DadosModel:
    _dados                      = None
    _atributoClassificador      = None

    def __init__(self, dados, atributoClassificador):
        self._dados = dados
        self._atributoClassificador = atributoClassificador