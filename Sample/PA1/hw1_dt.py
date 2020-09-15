import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        if len(self.features[0])==0:
            self.splittable=False
        if self.splittable:
            self.features=np.array(self.features)
            [r,c]=len(self.features),len(self.features[0]) 
            Gain=[]
            for j in range(c):
                #print("c",c)
                count=[]  
                for key in np.sort(np.unique(self.features[:,j])):
                    temp={}
                    for k in np.unique(self.labels):
                        temp[k]=0
                    for i in range(r):
                        if self.features[i][j]==key:
                            z=self.labels[i]
                            temp[z]=temp[z]+1
                    count.append(temp)
                sub = []
                for row in count:
                    row2 = []
                    for key in np.unique(self.labels):
                        row2.append(row[key])
                    sub.append(row2)
                Gain.append(Util.Information_Gain(0, sub))
            maxind=np.argwhere(Gain==np.max(Gain))
            if len(maxind)!=1:
                uft=[]
                for m in range(len(maxind)):
                    uft.append(len(np.unique(self.features[:,maxind[m][0]])))
                maxuft=np.argwhere(uft==np.max(uft))
                self.dim_split=maxind[maxuft[0][0]][0]
            else:
                self.dim_split=np.argmax(Gain)
            #print(self.dim_split)
            self.feature_uniq_split=np.sort(np.unique(self.features[:,self.dim_split]))
            for k in self.feature_uniq_split:
                feat_new=[]
                slce1=[]
                slce2=[]
                for i in range(r):
                    if self.features[i][self.dim_split]==k:
                        slce1=self.features[i]
                        slce1=np.delete(slce1,self.dim_split)
                        feat_new.append(slce1)
                        slce2.append(self.labels[i])
                child=TreeNode(feat_new,slce2,len(self.feature_uniq_split))
                self.children.append(child)
            for ch in self.children:
                ch.split()
        return
        raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        index=-1
        feature=np.array(feature)
        if self.splittable:   
            for k in range(len(self.feature_uniq_split)):
                if feature[self.dim_split]==self.feature_uniq_split[k]:
                    index=k
                    break
            if index!=-1:
                feature=np.delete(feature,self.dim_split)
                current=self.children[index]
                a=current.predict(feature)
        if index==-1 or not self.splittable:
            a=self.cls_max
        # feature: List[any]
        # return: int
        return a if True else 10**6
        raise NotImplementedError
