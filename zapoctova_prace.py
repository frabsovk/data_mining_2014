__author__ = 'Katerina'
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import csv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from standardise import standardise
import pylab as pl
from sklearn.metrics import confusion_matrix
import logging
import math
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from rdkit import DataStructs
import random


# funkce, ktera prevadi molekulu do standarniho tvaru
def standardizace (mol):
    parent = None
    try:
        parent = standardise.apply(mol)
        return parent
    except standardise.StandardiseException as e:
        logging.warn(e.message)

# funkce, ktera data rozdeluje na testovaci a trenovaci, uci model a provadi Gausianskou klasifikaci
def gausian(x, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y,
    test_size = 0.3)
    classifier = GaussianNB()
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    return y_pred,y_test

# funkce, ktera prevadi matici do obrazoveho formatu
def matrix_picture(matrix):
    pl.matshow(matrix)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()

# funkce pro vytvoreni grafu zavislosti predikovanych a testovanych hodnotot pEC50 pro ligandy
def regresion_plot(exper_data, predik_data,y_train,predTrain):

    # vytvoreni grafu
    plt.plot([2, 10], [2, 10], ls="--", c=".3")
    plt.plot(exper_data, predik_data, 'co', label='RBF model for testing data')
    plt.plot(y_train, predTrain,'co', c = 'r', label='RBF model for training data')

    #vypocet r2 grafu
    print('Pearsonuv korelacni koeficient modelu je roven ' + str(r2_score(exper_data,predik_data)))

    #vypocet RMSE grafu
    rms = sqrt(mean_squared_error(exper_data, predik_data))
    print('RMSE modelu je rovna ' + str(rms))

    # vypocet MAE grafu
    mae = mean_absolute_error(exper_data, predik_data)
    print('MAE modelu je rovna ' + str(mae))

    plt.ylim((2, 10))
    plt.xlabel('experimentalni_hodnoty pEC50')
    plt.ylabel('predikovane_hodnoty pEC50')
    plt.xlim((2, 10))
    plt.title('Support Vector Regression for ligand')
    plt.legend()
    plt.show()

# funkce pro rizeni zpracovani SVM regresni analyzy
def svm_count(ligand,decoy):

    # nadefinovani potrebnych promennych
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    decoy_test = []
    ligand_test = dict()

    print('SVM analyza k predikci pEC50 pro ligandy')
    print('\n')

    # rozdeleni dat na trenovaci a testovaci - testovaci = 30%, trenovaci = 70%
    test_ligand = random.sample(ligand.keys(), int(0.3*len(ligand)))
    for (key,val) in ligand.items():
        if key in test_ligand:
            ligand_test[key] = [val[1], val[0]]
        else:
            y_train.append(val[0])
            X_train.append(val[1])

    # Srovnani podobnosti testovacich dat s trenovacimi. Pokud je podobnost mensi nez 75%, nejsou tyto latky predikovany.
    vhodna_data = dict()
    for key,hodnoty in ligand_test.items():
        podobnost = []
        for data in X_train:
            podobnost.append(Similarity(hodnoty[0], data))
        if max(podobnost)> 0.75:
            vhodna_data[key] = [hodnoty[0], hodnoty[1]]
    for value in vhodna_data.values():
        X_test.append(value[0])
        y_test.append(value[1])

    print('Pocet molekul pro SVM analyzu: ' + str(len(ligand)))
    print('Pocet predikovanych hodnot po zohledneni podobnosti s trenovacmi daty: ' + str(len(X_test)))
    print('\n')

    # vybrani modelu pro svr analyzu a nastaveni parametru
    svr_rbf = SVR(kernel='rbf', degree=4, C=1e3, epsilon=0.2, gamma=.0001)

    # natrenovani modelu
    svr_rbf.fit(X_train, y_train)

    # pouziti modelu k predikci testovacich dat (predikce trenovacich dat poslouzi pri grafickem zpracovani analyzy)
    predSvr = svr_rbf.predict(X_test)
    predTrain = svr_rbf.predict(X_train)

    # zavolani funkce pro tvorbu grafu
    regresion_plot(y_test,predSvr,y_train,predTrain)

    print('\n')
    print('SVM analyza pro decoy molekuly: ')
    print('\n')

    # Srovnani podobnosti testovacich decoy molekul s trenovacimi daty. Pokud je podobnost mensi nez 80%, nejsou tyto latky predikovany.
    for hodnoty in decoy:
        podobnost = []
        for data in X_train:
            podobnost.append(Similarity(hodnoty, data))
        if max(podobnost)> 0.8:
            decoy_test.append(hodnoty)

    # predikce decoy
    predDecoy = svr_rbf.predict(decoy_test)
    print('Pocet predikovanych decoy molekul: ' + str(len(decoy_test)))
    print('Rozmezi hodnot pEC50 pro decoy je: ' + '( ' + str(min(predDecoy)) + ', ' + str(max(predDecoy)) + ')')

# funkce na vypocet podobnosti pomoci tanimotova koeficientu
def Similarity(test, train):
    similarity = DataStructs.FingerprintSimilarity(test, train, metric=DataStructs.TanimotoSimilarity)
    return similarity

# nadefinovani potrebnych datovych typu
ligand = dict()
spatny_standard = []
ligand_chyby = []
X = []
Y = []
svm_ligand = dict()
svm_decoy = dict()
svm_data = []
svm_target = []
svm_decoy_data = []

print("Program na predikci estrogenniho alfa receptoru pomoci Naivni Bayesovske klasifikace a SVM analyza pro predikci pEC50.")
print('\n')

# zpracovani sdf souboru s decoy molekulami
suppl = Chem.SDMolSupplier('er_agonist_decoys.sdf')
print('Zpracovavam soubor s decoy daty...')
print('\n')
for mol in suppl:
    smi = Chem.MolToSmiles(mol)
    m = Chem.MolFromSmiles(smi)

    # zaslani molekuly do funkce, ktera zajistuje standardizaci molekul
    standard_smi = standardizace(m)

    # zpracovani pouze standardnich molekul
    if standard_smi != None:
        fp = MACCSkeys.GenMACCSKeys(standard_smi)
        if smi not in svm_decoy_data:
            # ukladani hodnot do seznamu pro Bayesovskou analyzu
            X.append(fp)
            Y.append('decoy')
            # ukladani hodnot pro svm analyzu
            svm_decoy_data.append(fp)



print('Pocet decoy molekul vhodnych k analyze: ' + str(len(svm_decoy_data)))
print('\n')

pocet_molekul = 0;
data = csv.reader(file('data_chembl.xls'), dialect='excel', delimiter = '\t')
print('Zpracovavam soubor s daty z ChEMBLu...')
for row in data:
     pocet_molekul += 1
     # vyfiltrovani molekul, ktere maji nenulovou hodnotu EC50 v jednotkach nM
     if (row[10] !='' and row[12] == 'EC50' and row[15] == 'nM' ):
        ms = Chem.MolFromSmiles(row[10])

        # standardizace molekuly pomoci funkce standardiser
        y = standardizace(ms)
        if y != None:
            fps = MACCSkeys.GenMACCSKeys(y)
            # vyfiltrovani pouze unikatnich molekul, duplicitni molekuly ulozeny do seznamu ligand_chyby
            if (row[0] not in svm_ligand.keys() and row[0] not in ligand_chyby ):
                X.append(fps)
                Y.append('ligand')
                # ulozeni hodnot pEC50 a fps pro svm analyzu
                try:
                    svm_ligand[row[0]] = [- math.log10(float(row[14])*pow(10, -9)), fps]
                except (ValueError):
                    ligand_chyby.append(row[0])


            else:
                ligand_chyby.append(row[0])

# odtraneni duplicit z dat
for val in ligand_chyby:
    if val in svm_ligand.keys():
        svm_ligand.pop(val)

# vyfiltrovani molekul, ktere maji hodnotu pEC50 mensi nez 2.5 a vetsi nez 9
spatna_data = []
for (key,val) in svm_ligand.items():
    if (val[0] <= 2.5 or val[0] > 9):
        spatna_data.append(key)
for hodn in spatna_data:
    if hodn in svm_ligand.keys():
        svm_ligand.pop(hodn)


print('Pocet zpracovanych dat z ChEMBLu: ' + str(pocet_molekul))
print('Pocet molekul z ChEMBLu vhodnych k analyzam: ' + str(len(svm_ligand)))
print('\n')

# zaslani dat do funkce pro svm regresni analyzu
svm_count(svm_ligand,svm_decoy_data)

print('\n')
print('Analyza pomoci Naivniho Bayesovskeho klasifikatoru: ')
print('\n')

# volani funkce pro vytvoreni Bayesovskeho modelu k predikci ligandu a decoy
prediction, test = gausian(X, Y)

# Zpracovani a vypsani vysledku z Bayesovskeho modelu
target_names = ['decoy', 'ligand']
print('Vysledky klasifikace:\n')
print(classification_report(test, prediction, target_names=target_names))

# vypsani confusion matice
cm = confusion_matrix(test, prediction)
print(cm)

# volani funkce pro zobrazeni confusion matice v graficke podobe
matrix_picture(cm)




