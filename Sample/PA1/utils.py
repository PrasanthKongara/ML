import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    branches=np.array(branches)
    [r,c]=len(branches),len(branches[0])
    ent=0
    end=0
    for i in range(r):
        t=np.sum(branches[i])
        end=end+t
        add=0
        for j in range(c):
            p=float(branches[i][j])/t
            if p!=0:
                add=add-p*np.log2(p)
        ent=ent+add*t
    ent=float(ent)/end
    IG=S-ent
    return IG
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    raise NotImplementedError

def accu(node,X_test,y_test):
    test=[]
    for i in range(len(X_test)):
        test.append(node.predict(X_test[i]))
    acc=0
    for i in range(len(test)):
        if test[i]==y_test[i]:
            acc=acc+1
    return acc

def getnodes(node):
    res=[]
    for ch in node.children:
        if ch.splittable:
            res.append(ch)
            temp=getnodes(ch)
            if temp!=[]:
                for i in temp:
                    res.append(i)
    return res

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    node = decisionTree.root_node 
    next_iter=True
    while(next_iter):
        node.acc=accu(node,X_test,y_test)
        flag=False
        all_nodes=getnodes(node)
        all_nodes.append(node)
        best_acc=node.acc
        for i in range(len(all_nodes)):
            childs=all_nodes[i].children[:]
            all_nodes[i].children=[]
            all_nodes[i].splittable=False
            temp_acc=accu(node,X_test,y_test)
            if temp_acc>node.acc:
                if temp_acc>best_acc:
                    best_acc=temp_acc
                    flag=True
                    best_node=all_nodes[i]
            all_nodes[i].children=childs[:]
            all_nodes[i].splittable=True
            if flag:
                break
        if flag:
            best_node.children=[]
            best_node.splittable=False
            next_iter=True
        else:
            next_iter=False
    return
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    raise NotImplementedError



# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    real_labels,predicted_labels = np.array(real_labels),np.array(predicted_labels)
    a=np.sum(real_labels*predicted_labels)
    b=np.sum(predicted_labels)
    c=np.sum(real_labels)
    p=float(a)/b   
    r=float(a)/c
    score=2*p*r/(p+r)
    return score
    raise NotImplementedError

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    d = np.absolute(np.array(point1)-np.array(point2))
    dist=(np.sum(d*d))**0.5
    return dist
    raise NotImplementedError

#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(point1,point2)
    raise NotImplementedError



#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    d = np.absolute(np.array(point1)-np.array(point2))
    p = np.exp(-(np.sum(d*d))/2)
    return -p
    raise NotImplementedError


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    point1=np.array(point1)
    point2=np.array(point2)
    dist=float(np.inner(point1,point2))/((np.sum(point1*point1))**0.5 * (np.sum(point2*point2))**0.5)
    return 1-dist
    raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    Xtrain=np.array(Xtrain,dtype=float)
    ytrain=np.array(ytrain,dtype=int)
    Xval=np.array(Xval,dtype=float)
    yval=np.array(yval,dtype=int)
    f1=np.zeros((30,4))
    upper_k = 30
    if len(Xtrain) < 30:
        upper_k = len(Xtrain)
    m=0
    for k in range(1,upper_k,2):
        c=0
        for j in distance_funcs:         
            inst=KNN(k,distance_funcs[j])
            inst.train(Xtrain,ytrain)
            pred_val=inst.predict(Xval)
            f1[k][c]=f1_score(yval,pred_val) 
            if f1[k][c]>m:
                best_k=k
                best_func=j
                m=f1[k][c]
                best_model=inst
            c=c+1 
    print(f1)    
    print(best_model,best_k,best_func)
    return best_model,best_k,best_func
    raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    Xtrain=np.array(Xtrain,dtype=float)
    ytrain=np.array(ytrain,dtype=int)
    Xval=np.array(Xval,dtype=float)
    yval=np.array(yval,dtype=int)
    f1=np.zeros((30,4,2))
    upper_k = 30
    if len(Xtrain) < 30:
        upper_k = len(Xtrain)
    m=0
    for k in range(1,upper_k,2):
        c=0
        for j in distance_funcs:
            inst=KNN(k,distance_funcs[j])
            X_t=np.copy(Xtrain)
            X_v=np.copy(Xval)
            for i in scaling_classes:
                if i=='min_max_scale':
                    scale=MinMaxScaler()
                    Xtrain=scale.__call__(Xtrain)
                    c1=0
                    Xval=scale.__call__(Xval)
                if i=='normalize':
                    scale=NormalizationScaler()
                    Xtrain=scale.__call__(Xtrain)
                    c1=1
                    Xval=scale.__call__(Xval)
                inst.train(Xtrain,ytrain)
                pred_val=inst.predict(Xval)
                f1[k][c][c1]=f1_score(yval,pred_val) 
                if f1[k][c][c1]>m:
                    best_model=inst
                    best_k=k
                    best_func=j
                    best_scaler=i
                    m=f1[k][c][c1]
                Xtrain=np.copy(X_t)
                Xval=np.copy(X_v)
            c=c+1
    print(best_model,best_k,best_func,best_scaler)
    print(f1)
    return best_model,best_k,best_func,best_scaler    
    raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass
    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:  
        features=np.array(features,dtype=float)
        r=len(features)
        for i in range(r):
            v=features[i]
            dist=(np.sum(v*v))**0.5
            if dist!=0:
                features[i]=features[i]/dist
        return features                    
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.h=[]
        self.l=[]

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        features=np.array(features,dtype=float)
        if len(self.h)==0 and len(self.l)==0:
            h=np.amax(features, axis=0)
            l=np.amin(features, axis=0)
            self.h=h
            self.l=l
        else:
            h=self.h
            l=self.l
        [r,c]=len(features),len(features[0])
        for i in range(r):
            for j in range(c):
                if h[j]!=l[j]:
                    features[i][j]=(features[i][j]-l[j])/(h[j]-l[j])
                else:
                    features[i][j]=0
        return features 
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        raise NotImplementedError