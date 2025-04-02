import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

lengths = [len(item) for item in data_dict['data']]

# print(lengths)
# data = np.asarray(data_dict['data'],dtype=object)

min_length = min(lengths)
data_trimmed = [item[:min_length] for item in data_dict['data']]
lengths1 = [len(item) for item in data_trimmed]
# print(lengths1)
data = np.asarray(data_trimmed,dtype=object)

labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

precision = precision_score(y_test, y_predict, average='macro')  
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Classification Report:\n', classification_report(y_test, y_predict))

f = open('model.pickle', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("Model written in file")