from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from sklearn.preprocessing import Imputer
import copy

np.random.seed(1)

feature_names = ["Date","Location","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"]

data = np.genfromtxt('weatherAUS.txt', delimiter=',', dtype=str)
#labels = data[:,数据集attributes数量]
#le= sklearn.preprocessing.LabelEncoder()
#le.fit(labels)
#labels = le.transform(labels)
#class_names = le.classes_
#data = data[:,:-1]
from sklearn.preprocessing import Imputer

labels = data[:,22]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,:-1]

categorical_features = [0,1,7,9,10,21]
numerical_features = [2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

#for feature in numerical_features:
#    imp = Imputer(missing_values= 'NA', strategy='mean', axis=0)
#    print(data[:, feature].tolist)
#    imp.fit(data[:, feature].tolist())
#    data[:, feature] = imp.transform(data[:, feature])
#imp = Imputer(missing_values='NA', strategy='mean', axis=0)
#imp.fit(data)
#data = imp.transform(data)

data = data.astype(float)
print("check1")
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
print("check2")
encoder.fit(data)
encoded_train = encoder.transform(train)
#print("data0: ", data[0])
#print("data00: ", data[0][0])
#print("data0.shape: ",data[0].shape)
#print("reshapedata0: ",np.reshape(data[0],(1,22)))
#print("labels_train: ",labels_train)
#print("labels_test: ",labels_test)
#print("length: ",len(data))
#print("size: ",np.size(data))
#print("encoder: ",encoder.transform(data[0]))
#print("encoder2: ",encoder.transform(data[0].tolist()))
import xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)
#print("predict_proba: ",gbtree.predict_proba(np.reshape(data[0],(1,22))))
#gbtree.predict_proba(data[0])
print("Evaluate")
print(sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test))))

predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)
print("predict")
print(predict_fn)
explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names =
                                                   feature_names,class_names=class_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3)

print("check3")

minList = data[0].tolist()
maxList = data[0].tolist()
pivotList = data[0].tolist()
rangeList =data[0].tolist()

for i in range(1, len(data)):
    for ii in numerical_features:
        if (data[i][ii] < minList[ii]):
            minList[ii] = data[i][ii]
        if (data[i][ii] > maxList[ii]):
            maxList[ii] = data[i][ii]
            
for i in numerical_features:
    pivotList[i] = (maxList[i] + minList[i]) / 2
    rangeList[i] = maxList[i] - minList[i]

#    print("i: ", i)
#    print("min: ", minList[i])
#    print("max: ", maxList[i])
#    print("pivot: ", pivotList[i])
#    print("range: ", rangeList[i])

def cmp (a, b):
    if (a > b):
        return 1
    elif (a < b):
        return -1
    else:
        return 0

def takeSensi(elem):
    return elem[1]

def getFirst(elem):
    return elem[0]

sensitivity_count = 0

for i in range(1653, 1654):
    exp = explainer.explain_instance(data[i], predict_fn, num_features=15)
    exp.show_in_notebook(show_all=False)
    expdis = explainer.getDiscretized()
    for f in expdis.names:
        print("expdis_names[%d]: " % f,expdis.names[f], "min: ", minList[f], "max: ", maxList[f])

    print("i: ",i)
    print("exp2: ",exp.local_exp[1])
    #print("exp3: ",exp.local_exp[1][0])
    #print("exp4: ",exp.local_exp[1][0][1])
    print("blackbox: ",exp.predict_proba)
    #print("blackbox[0]: ",exp.predict_proba.tolist()[0]," blackbox[1]: ",exp.predict_proba.tolist()[1])

    feature_list = []
    sensitivity_list = []
    flag = cmp(exp.predict_proba.tolist()[0], exp.predict_proba.tolist()[1])

    for ii in range(0, len(exp.local_exp[1])):
        if((exp.local_exp[1][ii][1] >= 0.015 or exp.local_exp[1][ii][1] <= -0.015) and exp.local_exp[1][ii][0] in numerical_features):
            feature_list.append(exp.local_exp[1][ii][0])

    for ii in feature_list:
        print("ii: ", ii)

        test_data = copy.deepcopy(data[i])
        min_data = copy.deepcopy(data[i])
        min_data[ii] = minList[ii]
        max_data = copy.deepcopy(data[i])
        max_data[ii] = maxList[ii]

        min_exp = explainer.explain_instance(min_data, predict_fn, num_features=15)
        min_flag = cmp(min_exp.predict_proba.tolist()[0], min_exp.predict_proba.tolist()[1])

        max_exp = explainer.explain_instance(max_data, predict_fn, num_features=15)
        max_flag = cmp(max_exp.predict_proba.tolist()[0], max_exp.predict_proba.tolist()[1])

        if (flag == min_flag):
            left_rev = -1
        else:
            left_rev = 1

            upper = copy.deepcopy(test_data)
            lower = copy.deepcopy(min_data)

            while ((upper[ii] - lower[ii]) > (0.005 * rangeList[ii])):
                print("left: ", test_data[ii])
                test_data[ii] = (upper[ii] + lower[ii]) / 2
                test_exp = explainer.explain_instance(test_data, predict_fn, num_features=15)
                test_flag = cmp(test_exp.predict_proba.tolist()[0], test_exp.predict_proba.tolist()[1])
                if (test_flag == flag):
                    upper = copy.deepcopy(test_data)
                else:
                    lower = copy.deepcopy(test_data)

            left_range = abs(test_data[ii] - data[i][ii]) / rangeList[ii]

            for j in range(0, len(exp.local_exp[1])):
                if (exp.local_exp[1][j][0] == ii):
                    weight1 = exp.local_exp[1][j][1]                 
            for j in range(0, len(test_exp.local_exp[1])):
                weight2 = 0
                if (test_exp.local_exp[1][j][0] == ii):
                    weight2 = test_exp.local_exp[1][j][1]

            left_weight = abs(weight1 - weight2)
            left_sensi = left_weight / left_range

        if (flag == max_flag):
            right_rev = -1
        else:
            right_rev = 1
            upper = copy.deepcopy(max_data)
            lower = copy.deepcopy(test_data)

            while ((upper[ii] - lower[ii]) > (0.005 * rangeList[ii])):
                print("right: ", test_data[ii])
                test_data[ii] = (upper[ii] + lower[ii]) / 2
                test_exp = explainer.explain_instance(test_data, predict_fn, num_features=15)
                test_flag = cmp(test_exp.predict_proba.tolist()[0], test_exp.predict_proba.tolist()[1])
                if (test_flag == flag):
                    upper = copy.deepcopy(test_data)
                else:
                    lower = copy.deepcopy(test_data)

            right_range = abs(test_data[ii] - data[i][ii]) / rangeList[ii]

            for j in range(0, len(exp.local_exp[1])):
                if (exp.local_exp[1][j][0] == ii):
                    weight1 = exp.local_exp[1][j][1] 
            for j in range(0, len(test_exp.local_exp[1])):
                weight2 = 0
                if (test_exp.local_exp[1][j][0] == ii):
                    weight2 = test_exp.local_exp[1][j][1]
            right_weight = abs(weight1 - weight2)
            right_sensi = right_weight / right_range

        if (left_rev == -1 and right_rev == -1):
            sensitivity_list.append((ii, -1))
        elif (left_rev == 1 and right_rev == -1):
            sensitivity_list.append((ii, left_sensi))
        elif (left_rev == -1 and right_rev == 1):
            sensitivity_list.append((ii, right_sensi))
        elif (left_rev == 1 and right_rev == 1):
            sensitivity_list.append((ii, min(left_sensi, right_sensi)))     

    print(feature_list)
    sensitivity_list.sort(key = takeSensi, reverse = True)
    print(sensitivity_list)
    sensi_list = []
    for ii in range(0, len(sensitivity_list)):
        sensi_list.append(sensitivity_list[ii][0])

    print(sensi_list) 

    diff = 0
    for ii in range(0, len(feature_list)):
        feature_rank = ii
        sensi_rank = sensi_list.index(feature_list[ii])
        diff = diff + abs(feature_rank - sensi_rank)

    sensitivity_count = sensitivity_count + diff / len(feature_list)
    print("sensitivity_count: ", sensitivity_count)


'''
        if data[i][ii] < pivotList[ii]:
            for j in range(0, 11):
                #print("i: ", i,"ii: ", ii,"j: ", j)
                test_data = data[i]
                test_data[ii] = minList[ii] + rangeList[ii] * (j/10)
                test_exp =  explainer.explain_instance(test_data, predict_fn, num_features=15)
                for jj in range(0, len(test_exp.local_exp[1])):
                    if (test_exp.local_exp[1][jj][0] == ii):
                        sensitivity_list[ii].append((j, test_exp.local_exp[1][jj][1]))
                
        else:
            for j in range(0, 11):
                #print("i: ", i,"ii: ", ii,"j: ", j)
                test_data = data[i]
                test_data[ii] = maxList[ii] - rangeList[ii] * (j/10)
                test_exp =  explainer.explain_instance(test_data, predict_fn, num_features=15)
                for jj in range(0, len(test_exp.local_exp[1])):
                    if (test_exp.local_exp[1][jj][0] == ii):
                        sensitivity_list[ii].append((j, test_exp.local_exp[1][jj][1]))

    print(sensitivity_list)






np.random.seed(1)
numFidelity = 0
featureNum = 0
averageNum = 0.0
proFidelity = 0.0
for ii in range(1653, 1654):
    exp = explainer.explain_instance(data[ii], predict_fn, num_features=15)
    exp.show_in_notebook(show_all=False)
    expdis = explainer.getDiscretized()
    for f in expdis.names:
        print("expdis_names[%d]: " % f,expdis.names[f], "min: ", minList[f], "max: ", maxList[f])

    print("ii: ",ii)
    print("exp2: ",exp.local_exp[1])
    print("exp3: ",exp.local_exp[1][0])
    print("exp4: ",exp.local_exp[1][0][1])
    print("blackbox: ",exp.predict_proba)
    print("blackbox[0]: ",exp.predict_proba.tolist()[0]," blackbox[1]: ",exp.predict_proba.tolist()[1])




for ii in range(0,len(data)):
#ii = 1
#len(data)-1
    exp = explainer.explain_instance(data[ii], predict_fn, num_features=15)
    valuesum = 0
    print("currently: ",ii)
    #print("exp2: ",exp.local_exp[1])
    #exp.show_in_notebook(show_all=False)
    for iii in range(0, len(exp.local_exp[1])):
        #print("exp3: ",exp.local_exp[1][iii])
        if(exp.local_exp[1][iii][1] >= 0.015 or exp.local_exp[1][iii][1] <= -0.015):
            #print("iii: ",exp.local_exp[1][iii][1])
            valuesum = valuesum + exp.local_exp[1][iii][1]
            featureNum = featureNum + 1
    #print("value: ",valuesum)
    #print("proba: ",exp.predict_proba)
    if(exp.predict_proba[0] > exp.predict_proba[1]):
        if valuesum < 0:
            numFidelity = numFidelity + 1
    else:
        if valuesum > 0:
            numFidelity = numFidelity + 1

averageNum = featureNum/len(data)
proFidelity = numFidelity/len(data)
print("averageNum: ",averageNum)
print("probafid: ",proFidelity)
    
#ii = 1
#exp = explainer.explain_instance(data[ii], predict_fn, num_features=15)
#exp.show_in_notebook(show_all=False)

#for eachone in exp.local_exp[1]:
#ii = 1653
print("i",ii,"test[i]",data[ii])
#exp = explainer.explain_instance(data[ii], predict_fn, num_features=20)
#exp.show_in_notebook(show_all=False)

#ii = 10
#exp = explainer.explain_instance(data[ii], predict_fn, num_features=15)
#exp.show_in_notebook(show_all=False)

#ii = 2
#exp = explainer.explain_instance(data[ii], predict_fn, num_features=15)
#exp.show_in_notebook(show_all=False)

'''