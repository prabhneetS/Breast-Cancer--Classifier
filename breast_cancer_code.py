import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('breast_cancer_wisconsin.csv').values
X= dataset [:,:10]
Y= dataset [:, 10]
X_train , X_test , y_train , y_test = train_test_split (X , Y , test_size = 0.3 , random_state =42 )
from sklearn import tree
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , recall_score , f1_score , classification_report

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit( X_train , y_train )
def evaluation( y_test , y_pred ):
  accuracy = accuracy_score( y_test , y_pred )
  precision = precision_score( y_test , y_pred , average ='weighted')
  recall = recall_score( y_test , y_pred , average ='weighted')
  F1=( 2* precision * recall )/( precision + recall )
  return accuracy , precision , recall , F1
y_pred_X_test = classifier.predict( X_test )
y_pred_train =classifier.predict( X_train )
accuracy_test , precision_test , recall_test , F1_test= evaluation( y_test , y_pred_X_test )
accuracy1 , precision1 , recall1 , F1_1 = evaluation( y_train , y_pred_train )

print( f'Accuracy on the test set is {accuracy_test}, Accuracy on the train set is { accuracy1 }')
print( f'precision on the test set is { precision_test }, Precision on the train set is { precision1 }')

print( f'Accuracy on the test set is { recall_test }, Accuracy on the train set is {recall1 }')

print( f'Accuracy on the test set is {F1_test}, Accuracy on the train set is { F1_1 }')
options_of_titles = [
(" Confusion matrix , without normalization ", None ) ,
(" Normalized confusion matrix ", "true") ,
]
for title , normalize in options_of_titles :
  display = ConfusionMatrixDisplay.from_estimator(classifier , X_test , y_test , display_labels =['Benign','Malignant'],
  cmap = plt.cm.Blues ,
  normalize = normalize ,
  )
  display.ax_.set_title( title )
  print( title )
  print( display.confusion_matrix )

  criterion = ['gini','entropy','log_loss']
max_depth = [2 ,4 ,6 ,8 , 10 , 12]
results_obtained =[]
for crit in criterion :
  for dpth in max_depth :
    classifier = tree.DecisionTreeClassifier( criterion =crit ,
    max_depth = dpth )

    classifier = classifier.fit( X_train , y_train )
    y_pred_X_test =classifier.predict( X_test )
    accuracy_test , precision_test , recall_test , F1_test= evaluation ( y_test , y_pred_X_test )
    results_obtained.append([crit , dpth , accuracy_test , precision_test , recall_test , F1_test])
results_obtained =pd.DataFrame( results_obtained , columns =['Criterion','max_depth','Accuracy','Precision','Recall','F1'])

print('The best result in terms of F1 is:', results_obtained.iloc [ results_obtained ['F1']. argmax ()])
print('\n')
print('The best result in terms of Accuracy is:', results_obtained.iloc [ results_obtained ['Accuracy'].argmax ()])

print('\n')
print('The best result in terms of Precision is:', results_obtained.iloc [ results_obtained ['Precision'].argmax ()])

print('\n')
print('The best result in terms of Recall is:', results_obtained.iloc [ results_obtained ['Recall'].argmax ()])

plt.scatter ( results_obtained ['Criterion'], results_obtained ['F1'])
plt.title('F1 vs Criterion')
plt.ylabel('F1')
plt.xlabel('Criterion')
plt.show()
print('\n')
plt.scatter( results_obtained ['Criterion'], results_obtained ['Accuracy'])
plt.title('Accuracy vs Criterion')
plt.ylabel('Accuracy')
plt.xlabel('Criterion')
plt.show()
print('\n')
plt.scatter( results_obtained ['Criterion'], results_obtained ['Precision'])
plt.title('Precision vs Criterion')
plt.ylabel('Precision')
plt.xlabel('Criterion')
plt.show()
print('\n')
plt.scatter( results_obtained ['Criterion'], results_obtained ['Recall'])
plt.title('Recall vs Criterion')
plt.ylabel('Recall')
plt.xlabel('Criterion')
plt.show ()
print('\n')

plt.scatter( results_obtained ['max_depth'], results_obtained ['F1'])
plt.title('F1 vs Max Depth')
plt.ylabel('F1')
plt.xlabel('Max Depth')
plt.show()
print('\n')
plt.scatter( results_obtained ['max_depth'], results_obtained ['Accuracy'])
plt.title('Accuracy vs Max Depth')
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.show()
print('\n')
plt.scatter( results_obtained ['max_depth'], results_obtained ['Precision'])
plt.title('Precision vs Max Depth')
plt.ylabel('Precision')
plt.xlabel('Max Depth')
plt.show()
print('\n')
plt.scatter( results_obtained ['max_depth'], results_obtained ['Recall'])
plt.title('Recall vs Max Depth')
plt.ylabel('Recall')
plt.xlabel('Max Depth')