#------------------------------------------------------------------------------+
#
#   Morais, Kleyson.
#   April, 2018
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from Models import *
from EvaluationMetric.avaliador import AvaliadorController
import numpy as np
import random
import copy

class ParticulaController:

    dadosModel = None
    ac = None

    def __init__(self, avaliador):
        self.ac = avaliador

    def criarParticular(self, particula, dados):
        '''
        Esta função cria uma partícula para o enxame, personalizando a mesma para que tenha as características 
        do banco de dados informado.

        - São gerados aleatoriamente: Posição e Velociade
        - Cada partícula possui também a melhor posição pela qual ela já passou e seu respectivo fitness
        '''
        self.dadosModel = dados
        nLinhas, nAtributos = self.dadosModel._dados.shape
        #Criar array com posição (binário)
        particula._posicao = np.random.randint(2, size = nAtributos)
        #Criar array com velocidade (binário)
        particula._velocidade = np.random.randint(2, size = nAtributos)
        #Melhor posição já passada iniciar com a primeira posição
        # particula._melhorPosicaoLocal = particula._posicao
        #Salvar o fitness da respectiva posição
        self.atualizaFitness(particula)
        print("Partícula Criada:")
        print(particula._posicao,' | ', particula._fitness)

    def atualizaFitness(self, particula):
        '''
        Função para calcular o fitness da partícula, onde 1 significa atributo utilizado e 0 não utilizado
        '''        
        merito = self.ac.RandomForest(particula._posicao)
        if particula._fitness is None or merito > particula._fitness:
            particula._fitness = merito

    def atualizaPosicao(self, p1, p2, media_enxame):
        '''
        Esta função é responsável pela movimentação das partículas no espaço, calculando suas respectivas velocidades
        para descobrir as novas posições.

        - A variáveis c é constante para o cálculo, convencionalmente utiliza-se 2.5
        - e1 e e2 são variáveis de atrito para o movimento da partícula
        - valorMaximo é um limite que não permite a ser ultrapassada, convencionalmente utiliza-se [-6, 6]
        '''
        
        if p1._fitness > p2._fitness:
            pWin = p1
            pLoser = p2
        else:
            pWin = p2
            pLoser = p1
        
        c = 2.5
        #Número aleatório entre 0 e 1
        r1 = np.random.rand() 
        r2 = np.random.rand() 
        r3 = np.random.rand() 
        valorMaximo = 6

        for i, velocidade in enumerate(pLoser._velocidade):
            #Calculando velocidade
            velocidade = (r1 * velocidade) + (r2 * (pWin._posicao[i] - pLoser._posicao[i])) + (c * r3 * (media_enxame[i] - pLoser._posicao[i]))
            #Verificar Limite
            if abs(velocidade) > valorMaximo and abs(velocidade) is velocidade:
                velocidade = valorMaximo
            elif abs(velocidade) > valorMaximo:
                velocidade = -valorMaximo
            velocidade = self.sigmoid(velocidade)
            #Condicional de definição 0 ou 1
            if np.random.rand(1) < velocidade:
                pLoser._posicao[i] = 1
            else:            
                pLoser._posicao[i] = 0
        
        return pWin, pLoser
        
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-(x)))


class EnxameController:

    pc              = None
    dadosModel      = None
  
    def __init__(self, DadosModel, avaliador):
        self.pc = ParticulaController(avaliador)
        self.dadosModel = DadosModel

    def criarEnxame(self, enxame, nParticulas):
        print("Criando Enxame de Partículas")
        for i in range(nParticulas):
            #Criando instância de uma nova partícula e adicionando ao enxame
            novaParticula = ParticulaModel()
            self.pc.criarParticular(novaParticula, self.dadosModel)
            enxame._particulas.append(novaParticula)

        print("Enxame Criado Com Sucesso")
        
    def mediaEnxame(self, enxame):
        total = 0
        for particula in enxame._particulas:
            binario = ''
            for i, posicao in enumerate(particula._posicao):
                binario = binario + str(posicao)
            valor = int(binario, 2)
            total = total + valor
            tamanhoParticula = len(particula._posicao)

        media = int(total/len(enxame._particulas))
        binario = '{0:08b}'.format(media)
        charList = list(binario)

        if (tamanhoParticula - len(charList)) > 0:
            media_enxame = [0] * (tamanhoParticula - len(charList))
        else:
            media_enxame = []
        
        for i, bit in enumerate(charList):
            media_enxame.append(int(bit))
        return media_enxame
        
    def verificarMelhorPosicaoEnxame(self, enxame):
        print("Verificando Melhor Posição do Enxame")
        for particula in enxame._particulas:
            if (enxame._melhorFitness is None) or (particula._fitness > enxame._melhorFitness):
                enxame._melhorPosicaoGlobal = np.copy(particula._posicao)
                enxame._melhorFitness = particula._fitness    
            
        print("Partícula com a Melhor Posição Global: ")
        print(enxame._melhorPosicaoGlobal, ' | ', enxame._melhorFitness)

    def atualizaEnxame(self, enxame):
        
        enxameNovo = EnxameModel()
        # Selecionar duas Partículas do enxame
        media_enxame = self.mediaEnxame(enxame)

        while len(enxame._particulas) > 0:
            if len(enxame._particulas) >= 2:
                while True:
                    p1 = random.choice(enxame._particulas)
                    p2 = random.choice(enxame._particulas)
                    if p1 != p2:    
                        break
                
                pWin, pLoser = self.pc.atualizaPosicao(p1, p2, media_enxame)
                self.pc.atualizaFitness(pLoser)
                
                enxameNovo._particulas.append(pWin)
                enxameNovo._particulas.append(pLoser)

                enxame._particulas.remove(p1)
                enxame._particulas.remove(p2)
            else:
                resto = enxame._particulas.pop()
                enxameNovo._particulas.append(resto)

        for particula in enxameNovo._particulas:
            enxame._particulas.append(particula)
