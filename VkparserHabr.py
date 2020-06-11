import nltk
import vk
import json



session = vk.Session(access_token='a08952641f461dbaa9f99245b766d7a6896f18777baa353539a711333833e6492ebd1044e5b8dd1f33597')
vk_api = vk.API(session, v='5.103')


listofposts = []


#listofposts.append({'id':vk_api.wall.get(owner_id=-20629724)['id'],'date':j,'text':k,'friends':vk_api.friends.get(user_id=i)['items']})

#print(vk_api.wall.get(owner_id=-20629724,count = 100, filter = all, v = 5.103 ))

listofposts.append(vk_api.wall.get(owner_id=-20629724,count = 200, filter = all, v = 5.103 ))

listdatasethabr = []

for i in listofposts:
    listasist = []
    for j in i["items"]:
        print(j)
        listasist.append( {'id': j['id'], 'date': j['date'], 'text': j['text'], 'likes': j['likes']['count'],'reposts':j['reposts']['count'], 'views': j['views']['count']})
    listdatasethabr.append(listasist)


#for i in listdatasethabr:
    #print(len(i))
 #   with open("Habrdataset", "w", encoding="utf-8") as file:
  #      json.dump(i, file, ensure_ascii=False)




