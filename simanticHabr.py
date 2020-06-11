import re
import pymorphy2
from nltk.tokenize import sent_tokenize,word_tokenize
import nltk
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#import pandas as pd
#import networkx as nx
#import matplotlib.pyplot as plt
from colour import Color
import graphviz as gv
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
import csv
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



morph = pymorphy2.MorphAnalyzer()
vectorizer = TfidfVectorizer()
stop_word = stopwords.words('russian')
#stop_word_eng = stopwords.words('english')
stop_word.extend(['это', 'так', 'вот', 'быть', 'как', 'в', 'о', 'и', 'к', 'на', 'тот', 'при', 'от', 'до', 'с', 'из', 'не', '#','ха', 'is','the','a','an',
                  'are','in','for', 'of', 'ещё', 'уже', 'my','oh','ru','изз'])

list_of_friquancy = []
list_of_topic = []
list_of_friquancy2 = []
dict_topic = {}
list_of_likes = []
list_of_reposts = []


with open('C:/Users/Kristina/PycharmProjects/untitled6VKHabr/Habrdataset.json', 'r', encoding='utf-8') as fh: #открываем файл на чтение
    data = json.load(fh) #загружаем из файла данные в словарь data

## step 1 cleaning data

for i in data:
    cleaned_data = re.sub('[^А-Яа-я" "A-Za-z.,ё,й,ъ,ь,#]+','', i['text']).lower().replace(",", " ").replace(".", " ")
    tokens = cleaned_data.split()
    current_list = []
    cur_str = ""
    for j in tokens:
        if j not in stop_word:
            current_list.append(morph.parse(j)[0].normal_form)
            list_of_friquancy.append(morph.parse(j)[0].normal_form)
    #print(current_list)
    i['text'] = set(current_list)
    #list_of_tfidf.append(i['text'])

#Вычисление частотности слова

#text = nltk.Text(list_of_friquancy)
#text.plot(40)

for k in data:
    list_of_topic_current = []
    for h in k['text']:
        h = h.replace("#","")
        str_cur = ""
        for b in range(len(morph.parse(h))):
        #if ('LATN' or 'NOUN' or 'UNKN')  in morph.parse(h)[0].tag and len(h)>1:
        #    list_of_topic_current.append(h)
        #    #print(h)
            if 'LATN' in morph.parse(h)[b].tag and len(h) > 1:
                if h == "js":
                    h = 'javascript'
                if h == 'golang':
                    h = 'go'
                str_cur = h
                #list_of_topic_current.append(h)
                #list_of_friquancy2.append(h)
                #print(h)
            elif 'NOUN' in morph.parse(h)[b].tag :
                if h == 'стать' or h == 'изз':
                    break
                str_cur = h
                #list_of_topic_current.append(h)
                #list_of_friquancy2.append(h)
                #print(morph.parse(h))
            elif 'UNKN' in morph.parse(h)[b].tag:
                str_cur = h
                #list_of_topic_current.append(h)
                #list_of_friquancy2.append(h)
                #print(h)
            else:
                str_cur = ""
        if  str_cur != '':
            list_of_topic_current.append(str_cur)
            list_of_friquancy2.append(str_cur)
            #print(str_cur)
    list_of_topic.append(list_of_topic_current)
    k['text'] = list_of_topic_current
    dict_topic.update({k['id']:k['text']})

#for r in list_of_topic:
#    print(r)

#text = nltk.Text(list_of_friquancy2)
#text.plot(100)

list_of_experiment = list(dict_topic.values())

list_of_tween = list_of_experiment
list_intersection = []
list_of_object = []
list_of_concepts_current = []

count = 0
for loe in range(len(list_of_experiment)):
    for lot in range(len(list_of_tween)):
        str_list_intersection = []
        if list_of_experiment[loe] == list_of_tween[lot]:
            continue
        else:
            str_list_intersection = set(list_of_experiment[loe]).intersection(list_of_tween[lot])
            if str_list_intersection not in list_intersection:
                #print(str_list_intersection)
                list_intersection.append(str_list_intersection)


for q in list_intersection:
    list_of_object_cuttent = []
    for u,k in dict_topic.items():
        list_true = []
        for q1 in q:
            if q1 in k:
                list_true.append("True")
        if len(list_true) == len(q):
            list_of_object_cuttent.append(u)
    list_of_object.append(list_of_object_cuttent)

#### Compute likes

for loo in list_of_object:
    sum_of_likes_current = 0
    for e in data:
        for loo1 in loo:
            if e["id"] == loo1:
                sum_of_likes_current += e['likes']
    list_of_likes.append(sum_of_likes_current / len(loo))

####Compute reposts

for loo in list_of_object:
    sum_of_reposts_current = 0
    for e in data:
        for loo1 in loo:
            if e["id"] == loo1:
                sum_of_reposts_current += e['reposts']
    list_of_reposts.append(sum_of_reposts_current / len(loo))


####Concepts with all

for u in range(len(list_of_object)):
    list_of_concepts_current.append((list_of_object[u], list(list_intersection[u]),list_of_likes[u]))

#### Sortered concepts

cur_list_of_all_attr = []

for t1 in range(len(list_of_concepts_current)-1):
    for t2 in range(len(list_of_concepts_current)-t1-1):
        if len(list_of_concepts_current[t2][0]) < len(list_of_concepts_current[t2+1][0]):
            list_of_concepts_current[t2],list_of_concepts_current[t2+1] = list_of_concepts_current[t2+1],list_of_concepts_current[t2]


for h in list_of_concepts_current:
    for h1 in h[1]:
        if h1 not in cur_list_of_all_attr:
            cur_list_of_all_attr.append(h1)
            #print(h1)

list_of_concepts_current.append([[],cur_list_of_all_attr,51.15])


#for i in list_of_concepts_current:
#    print(i)

matrix = tuple(list_of_concepts_current)

## Calculate the groups

model = DBSCAN()
model2 = KMeans(n_clusters=2)
#max_value = max(i[2] for i in  matrix)
#print(max_value)
cur_cluster = []
for i in matrix:
    cur_cluster.append(i[2])

model.fit(np.reshape(cur_cluster,(-1, 1)))
model2.fit(np.reshape(cur_cluster,(-1, 1)))
all_predictions = model2.predict(np.reshape(cur_cluster,(-1, 1)))
n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)

#print(model.labels_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % list(model.labels_).count(-1))
print(model2.cluster_centers_)

color_groups = []

for i in all_predictions:
    if i == 0:
        color_groups.append("green")
    if i == 1:
        color_groups.append("red")
    if i == 2:
        color_groups.append("blue")



d = gv.Digraph(
    directory=None, edge_attr=dict(dir='none', labeldistance='1.5', minlen='2'))
red = Color("red")
file1 = open("concepts.txt", "a")
for m in matrix:
    #print(m)
    file1.write(str(m) + '\n')
file1.close()
matrixForGraph = []

for tup in matrix:
    if len([x for x in matrixForGraph if x[1] == tup[1]]) == 0:
        matrixForGraph.append(tup)
matrixForGraph = sorted(matrixForGraph, key=lambda x: x[2])
colors = list(red.range_to(Color("green"), len(matrixForGraph)))

color_counter = 0
for m in matrixForGraph:
    nodename = ', '.join([str(x) for x in m[1]])
    if matrixForGraph[0] == m:
        node_label = ' '
    else:
        node_label = nodename
    #print(nodename)
    d.node(nodename, node_label, color=color_groups[color_counter] , style='filled')
    color_counter += 1

    t = [x for x in matrixForGraph if set(m[1]).issubset(set(x[1])) and len(set(m[1])) != len(set(x[1]))]

    if len(t) > 0:
        all_neighbours = sorted(t, key=lambda x: len(x[1]))
        excluded_neighbours = []
        nearest_neighbours = all_neighbours
        for i in range(len(all_neighbours)):
            for j in range(i + 1, len(all_neighbours)):
                if i != j and set(all_neighbours[i][1]).issubset(set(all_neighbours[j][1])) and not all_neighbours[
                    j] in excluded_neighbours:
                    excluded_neighbours.append(all_neighbours[j])
        if len(excluded_neighbours) > 0:
            #print(excluded_neighbours)
            nearest_neighbours = [item for item in all_neighbours if item not in excluded_neighbours]

        for neighbour in nearest_neighbours:
            node_name2 = ', '.join([str(x) for x in neighbour[1]])
            d.edges([(node_label, node_name2)])

d.view()


'''

def recommendation():
    list_of_all_words = []

    list_of_items = []
    list_of_attr = []
    list_of_values = []

    for h1 in list_of_experiment:
        list_of_all_words.extend(h1)
    for i,j in dict_topic.items():
        print(i, j)
        for j1 in j:
            for l1 in list_of_all_words:
                if j1 == l1:
                    list_of_items.append(i)
                    list_of_attr.append(j1)
                    list_of_values.append(1)
                if j1 != l1:
                    list_of_items.append(i)
                    list_of_attr.append(j1)
                    list_of_values.append(0)

    #print(len(list_of_items),len(list_of_attr), len(list_of_values))

    dict_of_rec = []
    dict_of_rec.append(list_of_items)
    dict_of_rec.append(list_of_attr)
    dict_of_rec.append(list_of_values)
    #[list_of_items, list_of_attr,list_of_values]
    #print(dict_of_rec)
    for i in range(len(list_of_items)):
        print(list_of_items[i],list_of_attr[i],list_of_values[i])
    FILENAME = "recomendation_dataset.csv"
    #with open(FILENAME, "w", newline="") as file:
    #    writer = csv.writer(file)
    #    writer.writerows(dict_of_rec)




#recommendation()


'''


