import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

#il y a deux cas: 1. si le dataset contient de header, 2. le cas contraire

class Data_Cleaning:
    def __init__(self,csv_file, sep = ','):
        self.csv_file = csv_file
        self.df = None
        self.sep = sep
        self.cols = None
        self.x_num = None
        self.x_str = None
        self.x = None
        self.y = None
        self.target = None
        self.reply = None

    # est ce que le df contient de header ou non
    def isHeader(self):
        if self.reply == 'yes':
            self.df = pd.read_csv(self.csv_file, sep = self.sep)
        else:
            self.df = pd.read_csv(self.csv_file, header = None)
            self.cols = [c.strip() for c in self.cols.split(',')]
            self.df.columns = self.cols

    def separation_xnum_xstr(self):
        self.x_num = self.df.select_dtypes(include = [np.number])
        self.x_str = self.df.select_dtypes(exclude = [np.number])

        for i in self.x_str.columns:
            self.x_str[i] = self.x_str[i].fillna(self.x_str[i].mode()[0])

    def extraction_date(self):
        date_cols = []

        for col in self.x_str.columns:
            try:
                # Essayer de convertir la colonne en datetime
                self.x_str[col] = pd.to_datetime(self.x_str[col], infer_datetime_format=True, errors='coerce')
                date_cols.append(col)
            except:
                continue

        for col in date_cols:
            self.x_str[col + '_annee'] = self.x_str[col].dt.year
            self.x_str[col + '_mois'] = self.x_str[col].dt.month
            self.x_str[col + '_jour'] = self.x_str[col].dt.day
            self.x_str[col + '_jrsemaine'] = self.x_str[col].dt.weekday
            # Supprimer la colonne originale si nécessaire
            self.x_str.drop(columns=col, inplace=True)

    def encodage(self):
        encoder = LabelEncoder()

        for i in self.x_str.columns:
            self.x_str[i] = encoder.fit_transform(self.x_str[i])

        #Concatiner x_num et x_str
        self.x = pd.concat([self.x_num,self.x_str],axis = 1)

    def suppression_id(self):
        if self.target != '':
            self.x = self.x.drop(columns = self.target)

    # pour choisir la methode de remplissage des valeurs manquantes

    def val_manq(self):
        if self.reply == 'intelligente':
            impute = IterativeImputer(random_state = 42)
            self.x = pd.DataFrame(impute.fit_transform(self.x), columns = self.x.columns)
        elif self.reply == 'moyenne':
            self.x = self.x.fillna(self.x.mean())
        else:
            self.x = self.x.fillna(self.x.median())
    
    def duplication(self):
        self.x = self.x.drop_duplicates()

    def outlier(self,col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)

        IQR = Q3 - Q1

        Fb = Q1 - 1.5 * IQR
        Fh = Q3 + 1.5 * IQR

        return col.clip(lower = Fb, upper = Fh)
    
    def remp_outlier(self):
        for col in self.x.columns : 
            self.x[col] = self.outlier(self.x[col])

    # recuperation de nom de target

    def separation_x_y(self):
        self.y = self.x[self.target]
        self.x = self.x.drop(columns = self.target)
    

    #est ce que l'utilisateur souhaite de faire la standarisation ou non
    def standarisation(self):
        scale = StandardScaler()

        if self.reply == 'yes':
            self.x = pd.DataFrame(scale.fit_transform(self.x),columns = self.x.columns)
    
    def df_final(self):
        self.df = pd.concat([self.x,self.y], axis = 1)
        self.df.reset_index(drop = True)
        return self.df