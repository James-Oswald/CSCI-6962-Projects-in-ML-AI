
import pandas as pd
import numpy as np

#The feature list can be found in Tables 2 and 3 in the paper, 
#the first 14 are statistical features, the remaining are dynamic features
feature_names = [f"x_{i}" for i in range(51)]

#H1-H5 respectively correspond to heuristics 1-5 in the paper in Table 1.
heuristic_names = [f"H{x}" for x in range(1,6)]

#The final column indicates if a proof was found at all within reasonable time,
# we treat this as the 6th heuristic, H0
col_names = feature_names + heuristic_names + ["H0"] 

#The data is provided to us pre-split into train, test, and validation. 
#Since we will be doing k-fold validation later, we recombine it into a single dataframe. 
train_raw = pd.read_csv("./ml-prove/train.csv", names=col_names)
test_raw = pd.read_csv("./ml-prove/test.csv", names=col_names)
valid_raw = pd.read_csv("./ml-prove/validation.csv", names=col_names)
data_raw = pd.concat([train_raw, test_raw, valid_raw])

raw_copy = data_raw.copy()
heuristics = [f"H{i}" for i in range(6)]
for i,h in enumerate(heuristics):
    raw_copy[h] = (raw_copy[h] == 1).astype(int) * raw_copy[h] * i
raw_copy["H"] = raw_copy[heuristics].sum(axis=1)
raw_copy.drop(heuristics, axis=1, inplace=True)
data = raw_copy

from sklearn.model_selection import train_test_split
import random

train, test = train_test_split(data, test_size=0.05)

#print the accuracy on the train and test sets 
def score(model_name, model):
    print(model_name + " Train Accuracy Score: %.2f" % model.score(train[feature_names], train["H"]))
    print(model_name + " Test Accuracy Score:  %.2f" % model.score(test[feature_names], test["H"]))

from sklearn.base import BaseEstimator, ClassifierMixin

class MyDecisionTreeNode:
    def __init__(self, feature=None, threshold=None, dtClass=None, left=None, right=None):
        self.feature = feature      #Index into the feature split on this node
        self.threshold = threshold  #Value at which the feature splits
        self.dtClass = dtClass      #If leaf node, this is set to the size
        self.left = left            #left sub decision tree
        self.right = right          #right sub decision tree
        

class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin): #custom sklearn estimator class
    def __init__(self, max_depth=np.Infinity, min_sample_split=1, max_features=None):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

        #hyperparameters
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        np.seterr(divide='ignore', invalid='ignore')
        
        self.root = None
        self.n_classes = None
    
    def _evaluate_numeric_attribute(self, X, y, feature):
        X_feature = X[feature].tolist()
        xy = list(zip(X_feature, y))
        xy.sort(key=lambda x:x[0])
        X_sorted, y_sorted = zip(*xy)

        #compute list of potential split points
        midpoints = []
        ni = np.zeros(self.n_classes)
        for j in range(X.shape[0] - 1):
            ni[y_sorted[j]] += 1   #running class count
            if X_sorted[j] != X_sorted[j+1]:
                v = (X_sorted[j] + X_sorted[j+1]) / 2
                Nv = np.copy(ni)
                midpoints.append((v, Nv, j))
        
        #iterate over our midpoints to find which one has the best Gini score
        (best_threshold, best_score) = (None, 0)
        n = X.shape[0]
        for (v, Nv, nearest) in midpoints:
            PciDy, PciDn = np.zeros(self.n_classes), np.zeros(self.n_classes) #probabilities of positive class selection (PciDy) and negative (PciDN)
            PciDy = Nv / np.sum(Nv)
            PciDn = (ni - Nv) / np.sum(ni - Nv)  
            nn = nearest
            ny = n - nearest

            #for our scoring we use gini, which is what sklearn uses in their decision tree
            Gini_index = (ny/n)*(1-np.sum(PciDy**2)) + (nn/n)*(1-np.sum(PciDn**2))
            if Gini_index > best_score:
                (best_threshold, best_score) = (v, Gini_index)
        
        return (best_threshold, best_score)


    def _build_tree(self, X, y, depth):
        print("building at depth" + str(depth))
        n = X.shape[0]
        ni = np.array([(y==k).sum() for k in range(self.n_classes)])   #sizes of each class i
        
        #base case, create a leaf node
        if depth >= self.max_depth or n <= self.min_sample_split:
            dtClass = np.argmax([ni[i] / n for i in range(self.n_classes)])   #set the leaf class as the majority class
            return MyDecisionTreeNode(dtClass=dtClass)

        #recursive case
        else:
            #Find the ideal feature and threshold to split our tree at
            (best_threshold, best_feature, max_score) = (None, None, -np.Infinity)
            features_sampled = 0
            features = set(feature_names.copy())
            #iterate until we sample all features or find at least 1 useable threshold 
            while(features_sampled < self.max_features or best_threshold == None):
                if(features_sampled >= self.max_features and best_threshold == None): #give up because no threshold was found
                    dtClass = np.argmax([ni[i] / n for i in range(self.n_classes)])   #set the leaf class as the majority class
                    return MyDecisionTreeNode(dtClass=dtClass)
                features_sampled += 1
                feature = random.sample(features, 1)[0]
                features.remove(feature)
                (threshold, threshold_score) = self._evaluate_numeric_attribute(X, y, feature) #all of our attributes are numeric
                if threshold_score > max_score:
                    (best_threshold, best_feature, max_score) = (threshold, feature, threshold_score)
            
            #Split the data row by row
            Xy_array, Yy_array, Xn_array, Yn_array  = [], [], [], []
            for (index, row), label in zip(X.iterrows(), y.array):
                if row[best_feature] <= best_threshold:
                    Xn_array.append(row)
                    Yn_array.append(label)
                else:
                    Xy_array.append(row)
                    Yy_array.append(label)
            Xy, Yy, Xn, Yn = pd.DataFrame(Xy_array), pd.Series(Yy_array), pd.DataFrame(Xn_array), pd.Series(Yn_array)

            #recursively fill out the rest of the tree
            root = MyDecisionTreeNode()
            root.feature = best_feature
            root.threshold = best_threshold
            root.left = self._build_tree(Xn, Yn, depth+1)
            root.right = self._build_tree(Xy, Yy, depth+1)
            return root
            

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.max_features = X.shape[1]  #sample all features
        self.root = self._build_tree(X, y, 0)
        return self
    
    def _tree_predict(self, X, root):

        #base case, the node is a leaf with a class prediction
        if root.dtClass != None:
            return root.dtClass

        #recursive case, check if above threshold to determine direction of decent 
        else:
            if X[root.feature] <= root.threshold:
                return self._tree_predict(X, root.left)
            elif X[root.feature] > root.threshold:
                return self._tree_predict(X, root.right)


    def predict(self, X):
        rv = []
        for index, row in X.iterrows():
            rv.append(self._tree_predict(row, self.root))
        return np.array(rv)

