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


print( Chem.MolFromMol2File('er_agonist_decoys.mol2'))
#ps = tuple(MACCSkeys.GenMACCSKeys(p))


#for mol in p:
#     print(mol)

data = csv.reader(file('data_chembl.xls'),dialect='excel',delimiter = '\t')

trideni = dict()
radek =0
chyba = set()
for row in data:
     if (row[10] !='' and row[12] == 'EC50' and row[15] == 'nM' ):

        ms = Chem.MolFromSmiles(row[10])
       # print(radek)
        fps = tuple(MACCSkeys.GenMACCSKeys(ms))
        if row[0] not in trideni.keys():
            trideni[row[0]] = [row[10],row[12],row[15],'ligand',fps]
            radek+=1
        else:
            chyba.add(row[0])
            #print(row)

            # print (row[12],row[15]

for key,val in trideni.items():
    print (key,val)

#print(len(trideni))

