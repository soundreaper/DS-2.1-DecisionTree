import io
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.tree import export_graphviz

data = pd.read_csv('tenni.txt', delimiter="\t", header=None, names=['a', 'b', 'c', 'd', 'e'])
data_encoded = data.apply(preprocessing.LabelEncoder().fit_transform)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(data_encoded[['a', 'b', 'c', 'd']], data_encoded['e'])
dot_data = export_graphviz(clf, out_file="tree.dot", feature_names=['Outlook', 'Temp.', 'Humidity', 'Wind'])
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_png('tennis_tree.png')