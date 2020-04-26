import NN
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp

import tensorflow as tf

from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer





class CompletData:
    def __init__(self, data_path, primary_key, final_target):
        self.data_path = data_path
        self.df_ori = pd.read_csv(data_path)
        self.primary_key = primary_key
        self.final_target = final_target
       

    def importantCoeffFinalTarget(self, number_to_keep = 30):
        """
        Ridge regression, feature selection for final target

        :return: Nothing. modify self.df_ori
        """
        inter = self.extractXY_values(self.df_ori, self.final_target)
        x = inter[0]
        y = inter[1]
        columns_entry = inter[2]
        coefs = self.importantCoeff(x, y, number_to_keep, columns_entry, alpha = 12)
        
        return(coefs)

        

    def fillColumns(self, column_names):
        """
        Fill the NaN for one column
        :param columns: column names to fill up
        :return: Nothing. modify self.df_ori
        """
        n = len(column_names)
        count = 1
        for column in column_names:
            print(column)
            self.oneColumn(column)
            tf.reset_default_graph()
            print(column + " " + str(count)+ "/" + str(n) + " is completed")
            count += 1

    def extractXY_values(self, df, name_y):

        df = df.fillna(0)
        columns_entry = []
        for a in range(len(df.columns)):
            if df.columns[a] != name_y and df.columns[a] != self.final_target and df.columns[a] != self.primary_key:
                columns_entry.append(df.columns[a])
                      
        x = df.loc[:, columns_entry]
        y = df.loc[:,[name_y]]

        x_values = x.values
        y_values = y.values

        # Normalisation. StandardScaler may not be appropriate
        sc = StandardScaler().fit(x_values)
        X = sc.transform(x_values)
        Y = y_values

        return(X,Y, columns_entry)

        

    def oneColumn(self, name_column):
        """
        Fill the NaN for one column
        :param df: dataframe
        :param name_column: column name to fill up
        :return: Nothing. Modify self.df_ori
        """

        df = self.df_ori[self.df_ori[name_column].notnull()]
        df = df.fillna(0)
        inter = self.extractXY_values(df, name_column)
        X = inter[0]
        Y = inter[1]
        columns_entry = inter[2]
##        print(columns_entry)

        # Extract important coefficients. Just ridge regression
        columns_entry = self.importantCoeff(X, Y, 12, columns_entry, alpha = 12)

        x_values  = df.loc[:, columns_entry].values
        y_values  = df.loc[:, [name_column]].values
              
        # Normalisation. MinMaxScaler may not be appropriate
        mmsx =  MinMaxScaler().fit(x_values)
        X = mmsx.transform(x_values)
        mmsy =  MinMaxScaler().fit(y_values)
        Y = mmsy.transform(y_values)
                   
        # handle data missing values 
        df_cor = self.df_ori[self.df_ori[name_column].isnull()]
        len_df_cor = len(df_cor)
        len_df_ori = len(self.df_ori)
##        print(len_df_cor/len_df_ori)
        if len_df_cor!=0 and len_df_cor/len_df_ori<0.2:
            df_cor = df_cor.fillna(0)
            x_cor = df_cor.loc[:, columns_entry]
            y_cor = df_cor.loc[:, [name_column]]
            x_cor_values = x_cor.values
            y_cor_values = y_cor.values
            X_cor = mmsx.transform(x_cor_values)
            Y_cor = mmsy.transform(y_cor_values)

            # give the mean value to, to avoid bias. Only for RBM or Autoencoder
            Y_cor = self.meanValueToNaN(Y, Y_cor)

            # Model learning
            nn = NN.nn(X.shape[1], [10], learning_rate = 0.01)
            nn.train(X, Y,  X_cor, batch_size = 40, n_epoches = 15)

            # inverse scaler
            corrected_column = mmsy.inverse_transform(nn.test)
            
            # correct columns in original dataframe
            self.fillDfOneColumn(df_cor, corrected_column, name_column)
                   
    def fillDfOneColumn(self, df_cor_i, correct_column, name_column):
        """
        Fill up the NaN of one Column in the self.
        
        :param df_cor_i: piece of dataframe containing missing value for the column $name_column
        :param correct_column: Values that will replace the NaN
        :param name_column: Column of interest
        :return: does not return anything. Simply modify self.df_ori
        """

        # slow version
##        for a in range(len(self.df_ori)):
##            inte = df_cor_i[[self.primary_key]].values
##            if self.df_ori.loc[a,[self.primary_key]].values in inte:
##                indi = 0
##                verif = False
##                while indi < len(inte) and verif == False:
##                    if inte[indi] == self.df_ori.loc[a,[self.primary_key]].values:
##                        self.df_ori.loc[a,[name_column]] = correct_column[indi]
##                        verif = True
##                    indi += 1

        
        for a in range(len(df_cor_i)):
            self.df_ori[name_column][self.df_ori[self.primary_key] == df_cor_i.loc[df_cor_i.index[a],[self.primary_key]].values[0]] = correct_column[a]


                        
    def importantCoeff(self, x, y, number_to_keep, columns_entry, alpha = 12):
        """
        Get $number_to_keep most important coefficients
        :param x: features
        :param y: target
        :param number_to_keep: number of coefficient to keep
        :return: List of String. Name of the column to keep ranked from most important to less important
        """
        # Ridge regression
        ridge = Ridge(alpha = 12)
        ridge.fit(x,y)

        zip_coefName = zip(np.abs(ridge.coef_.flatten()),columns_entry)
        listColKept = []
        for a in zip_coefName:
            listColKept.append([a[0],a[1]])

##        print(listColKept)

        listColKept.sort()
        listColKept.reverse()
##        print(listColKept)

        columns_entry = []
##        print(number_to_keep)
        for a in range(number_to_keep):
            print(a)
            columns_entry.append(listColKept[a][1])

        return(columns_entry)
            
    def meanValueToNaN(self, y_train, y_nan):
        y_train_cp = y_train.copy()
        y_nan_cp = y_nan.copy()
        moy = 0
        c = 0
        for a in range(len(y_train_cp)):
            moy += y_train_cp[a][0]
            c+=1
        moy = moy/c
        for a in range(len(y_nan_cp)):
            y_nan_cp[a][0] = moy

        return(y_nan_cp)


    def export(self, export_data_path):
        self.df_ori.to_csv(export_data_path)


        
















                      
