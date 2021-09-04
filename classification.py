#######################################################################
# Classification Exploration
# Author: Lindy Bustabad, with code from Dr. Christopher Healey
# Date: September 4, 2021
# Website for code: https://www.csc2.ncsu.edu/faculty/healey/msa/python/
# 
# Exploration of sklearn supervised classification techniques: 
#   1) Perceptron (neural net)
#   2) SVC (support vector classification)
#   3) KNN (k nearest neighbors)
#   4) Random forest (ensemble decision tree approach)
# 
# Banknote Authentication Data Set: Machine Learning Repository, UCI Center for 
# Machine Learning and Intelligent Systems
# Website: https://archive.ics.uci.edu/ml/datasets/banknote+authentication#
# 
# Data set description: 
#   Data were extracted from images that were taken from 
#   genuine and forged banknote-like specimens. For digitization, an industrial camera
#   usually used for print inspection was used. The final images have 400x 400 pixels. 
#   Due to the object lens and distance to the investigated object gray-scale pictures 
#   with a resolution of about 660 dpi were gained. Wavelet Transform tool were used 
#   to extract features from images.
# 
# Data set citation:
#   Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
#       [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
#       School of Information and Computer Science.
#
# Owner of database: Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)
# Donor of database: Helene DÃrksen (University of Applied Sciences, Ostwestfalen-Lippe, helene.doerksen '@' hs-owl.de)
#######################################################################

import numpy as np
import sklearn
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

banknote = pandas.read_csv('data_banknote_authentication.csv', encoding='latin1')
print(banknote.describe())
print(banknote.shape)

banknote.hist(figsize = (10, 10))
plt.show()

X = banknote.drop('class', axis = 1)
Y = banknote['class']

print(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y)

print( 'Number of training samples:', x_train.shape[ 0 ] )
print( 'Number of test samples:', x_test.shape[ 0 ] )

uniq, n = np.unique( Y, return_counts = True )
print( 'Frequency of class values in y:', uniq, n )
uniq, n = np.unique( y_train, return_counts = True )
print( 'Frequency of class values in training dataset:', uniq, n )
uniq, n = np.unique( y_test, return_counts = True )
print( 'Frequency of class values in test dataset:', uniq, n )

# Fit a standard scalar to the predictive variables
sc = StandardScaler()
sc.fit( x_train )

# Transform predictive datasets based on the scalar
X_train_std = sc.transform( x_train )
X_test_std = sc.transform( x_test )

# Join (stack) the data training and test sets into a sequence of rows
X_comb_std = np.vstack( ( X_train_std, X_test_std ) )

# Join (stack) the target training and test sets into a 1D horizontal column
y_comb = np.hstack( ( y_train, y_test ) )

# Print some descriptive statistics about training and target
print( 'Standardized data array:' )
print( X_comb_std[ :5 ], '\n---\n', X_comb_std[ -5: ] )
print( 'Target array:' )
print( y_comb )

# Build a perceptron model based on training data, then classify test data
ppn = Perceptron( max_iter=40, eta0=0.1, random_state=0 )
ppn.fit( X_train_std, y_train )
y_pred = ppn.predict( X_test_std )

# Print results of classification
print( 'Perceptron Model:' )
print( 'Misclassified samples: %d' % ( y_test != y_pred ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( ppn.score( X_train_std, y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( ppn.score( X_test_std, y_test ) ) )

# Build an SVC model based on training data, then classify test data
svc = SVC( kernel='rbf', gamma=0.1, C=1.0, random_state=0 )
svc.fit( X_train_std, y_train )
y_pred = svc.predict( X_test_std )

# Print results of classification
print( 'SVC Model:' )
print( 'Misclassified samples: %d' % ( y_test != y_pred ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( svc.score( X_train_std, y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( svc.score( X_test_std, y_test ) ) )

# Build a KNN model based on training data, then classify test data
knn = KNeighborsClassifier( n_neighbors=5, p=2, metric='minkowski' )
knn.fit( X_train_std, y_train )
y_pred = knn.predict( X_test_std )

# Print results of classification
print( 'KNN Model:' )
print( 'Misclassified samples: %d' % ( y_test != y_pred ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( knn.score( X_train_std, y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( knn.score( X_test_std, y_test ) ) )

# Build a random forest model based on training data, then classify test data
rf = RandomForestClassifier( n_estimators=100, oob_score=True )
rf.fit( X_train_std, y_train )
y_pred = rf.predict( X_test_std )

# Print results of classification
print( 'Random Forest Model:' )
print( 'Misclassified samples: %d' % ( y_test != y_pred ).sum() )
print( 'Accuracy on training data: {:.2f} (out of 1)'.format( rf.score( X_train_std, y_train ) ) )
print( 'Accuracy on test data: {:.2f} (out of 1)'.format( rf.score( X_test_std, y_test ) ) )
print( 'Out-of-bag accuracy: {:.2f} (out of 1)'.format( rf.oob_score_ ) )