import numpy as np

from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict

class AvaliadorController():

    dados                   = None
    atributoClassificador   = None

    def __init__(self, dados, atributoClassificador):
        self.dados                  = dados
        self.atributoClassificador  = atributoClassificador

    def selectionFeatures(self, features):
        aux = np.asarray(self.dados)
        if np.count_nonzero(features) == 0:
            X_subset = aux
        else:
            X_subset = aux[:,features==1]
        return X_subset

    def allClassificadores(self):
        ac1 = self.metricas(RandomForestClassifier(random_state=43))
        print('Random Forest: Acurácia usando todos os atributos (', self.dados.shape[1] ,' atributos) é: ', ac1*100,'%')

        ac2 = self.metricas(GaussianNB())
        print('Naive Bayes: Acurácia usando todos os atributos (', self.dados.shape[1] ,' atributos) é: ', ac2*100,'%')
        
        ac3 = self.metricas(tree.DecisionTreeClassifier())
        print('Decision Tree: Acurácia usando todos os atributos (', self.dados.shape[1] ,' atributos) é: ', ac3*100,'%')
        
        # ac4 = self.metricas(svm.SVC(kernel='linear', C=1))
        # print('SVM: Acurácia usando todos os atributos (', self.dados.shape[1] ,' atributos) é: ', ac4*100,'%')

    def allClassifiers(self, features):
        X_subset = self.selectionFeatures(features)

        print("\nRandom Forest Classifier")        
        f1score, acuracia = self.allMetrics(RandomForestClassifier(random_state=43), X_subset=self.dados)
        print('---------------------------------------------------------------------------------------')
        print('F1 Score na base original (', self.dados.shape[1] ,' atributos) é: ', f1score*100,'%')
        print('Acurácia na base original (', self.dados.shape[1] ,' atributos) é: ', acuracia*100,'%')
        print('---------------------------------------------------------------------------------------')
        f1score, acuracia = self.allMetrics(RandomForestClassifier(random_state=43), features=features)
        print('F1 Score após a Selection Feature Aplicada (', X_subset.shape[1] ,' atributos) é: ', f1score*100,'%')
        print('Acurácia após a Selection Feature Aplicada (', X_subset.shape[1] ,' atributos) é: ', acuracia*100,'%')
        print('---------------------------------------------------------------------------------------')

        # print("\nNaive Bayes Classifier")
        # ac2 = self.NaiveBayes(features)
        # print('Acurácia após a Selection Feature Aplicada (', X_subset.shape[1] ,' atributos) é: ', ac2*100,'%')

        # print("\nDecision Tree Classifier")
        # ac3 = self.DecisionTree(features)
        # print('Acurácia após a Selection Feature Aplicada (', X_subset.shape[1] ,' atributos) é: ', ac3*100,'%')

        # print("\nSupport Vector Machines Classifier")
        # ac4 = self.SupportVectorMachines(features)
        # print('Acurácia após a Selection Feature Aplicada (', X_subset.shape[1] ,' atributos) é: ', ac4*100,'%')

    def RandomForest(self, features):
        # print("\nRandom Forest Classifier")        
        return self.metrics(RandomForestClassifier(random_state=43), features)

    def NaiveBayes(self, features):
        # print("\nNaive Bayes Classifier")
        return self.metrics(GaussianNB(), features)

    def DecisionTree(self, features):
        # print("\nDecision Tree Classifier")
        return self.metrics(tree.DecisionTreeClassifier(random_state=43), features)

    def SupportVectorMachines(self, features):
        # print("\nSupport Vector Machines Classifier")
        return self.metrics(svm.SVC(kernel='linear', C=1), features)

    def metrics(self, classificador, features):
        X_subset = self.selectionFeatures(features)
        predicao = cross_val_predict(classificador, X_subset, self.atributoClassificador, cv=10)
        f1Score = f1_score(self.atributoClassificador, predicao, average='macro')
        return f1Score

    def allMetrics(self, classificador, features=None, X_subset=None):
        if X_subset is None:
            X_subset = self.selectionFeatures(features)
        predicao = cross_val_predict(classificador, X_subset, self.atributoClassificador, cv=10)
        f1Score = f1_score(self.atributoClassificador, predicao, average='macro')
        acuracia = accuracy_score(self.atributoClassificador, predicao)
        return f1Score, acuracia

    