__author__ = 'Katerina'
import numpy
from rdkit import Chem
#from rdkit.Chem import rdMolDescriptors as desc
from rdkit.Chem import MACCSkeys
import scipy
import sklearn
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import pprint
import standardise
import pylab as pl
from sklearn.metrics import confusion_matrix

finalnidata = []

decoy = {}
chyby = []
X = []
Y = []
suppl = Chem.SDMolSupplier('er_agonist_decoys.sdf')
id = 0
for mol in suppl:
    smi = Chem.MolToSmiles(mol)
    m = Chem.MolFromSmiles(smi)
    fp = tuple(MACCSkeys.GenMACCSKeys(m))
    if smi not in decoy.keys():
        X.append(fp)
        Y.append('decoy')
        finalnidata.append(['decoy',fp])
        decoy[smi] = [id,smi,'decoy',fp]
        id+=1
    else:
        chyby.append(smi)

#for key,val in decoy.items():
   # print(key,val)

#print(len(chyby))

data = csv.reader(file('data_chembl.xls'),dialect='excel',delimiter = '\t')

trideni = dict()
radek =0
chyba = set()
svm = dict()
for row in data:
     if (row[10] !='' and row[12] == 'EC50' and row[15] == 'nM' ):

        ms = Chem.MolFromSmiles(row[10])
       # print(radek)
        fps = tuple(MACCSkeys.GenMACCSKeys(ms))
        if row[0] not in trideni.keys():
            X.append(fps)
            Y.append('ligand')
            finalnidata.append(['ligand',fps])
            trideni[row[0]] = [row[10],row[12],row[15],'ligand',fps]
            radek+=1
        else:
            chyba.add(row[0])
            #print(row)

            # print (row[12],row[15]

#for key,val in trideni.items():
 #   print (key,val)

#print(len(decoy))
#print(len(trideni))
#print(len(finalnidata))
spravne_zarazeni = 0
spatne_zarazeni = 0
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y,
test_size = 0.3)
classifier = GaussianNB()
y_pred = classifier.fit(X_train, y_train).predict(X_test)
y_predictions = classifier.predict(X_test)

#print(len(y_predictions),len(y_test))

cm = confusion_matrix(y_test, y_pred)

print(cm)

pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()