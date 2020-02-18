import pandas as pd
import numpy as np
import math

moviesDF = pd.read_csv("G:/RecommenderSystem/movieslens/ml-latest-small/movies.csv")
usersDF = pd.read_csv("G:/RecommenderSystem/movieslens/ml-latest-small/ratings.csv")


#利用pandas的数据透视功能，得到用户-电影评分矩阵
ratingMatrix = pd.pivot_table(usersDF[['userId', 'movieId', 'rating']], index=['userId'], columns=['movieId'], values=['rating'],fill_value=0)

ratings = ratingMatrix.values.tolist()
userMap = dict(enumerate(list(ratingMatrix.index)))
moviesMap = dict(enumerate(list(ratingMatrix.columns)))

#利用余弦相似度计算用户相似度
def calCosineSim(list1, list2):
    val, list11, list22 = 0, 1, 1
    for (a, b) in zip(list1, list2):
        val += a * b
        list11 += a**2
        list22 += b**2
    return val/math.sqrt(list11*list22)

userSim = np.zeros((len(ratings), len(ratings)), dtype=np.float32)
for i in range(len(ratings)-1):
    for j in range(i+1, len(ratings)):
        userSim[i][j] = calCosineSim(ratings[i], ratings[j])
        userSim[j][i] = userSim[i][j]

#提取和用户兴趣最近的10个用户
userSimHigh = dict()
for i in range(len(ratings)):
    userSimHigh[i] = sorted(enumerate(list(userSim[i])), key = lambda x:x[1], reverse = True)[:10]

#用户对其未看过的电影的兴趣度矩阵
movieRecom = np.zeros((len(ratings), len(ratings[0])), dtype=np.float32)
for i in range(len(ratings)):
    for j in range(len(ratings[0])):
        if ratings[i][j]==0:
            for user,sim in userSimHigh[i]:
                movieRecom[i][j] += sim * ratings[user][j]

#提取用户最感兴趣的、其之前未看过的10部电影
movieRecomHigh = dict()
for i in range(len(ratings)):
    movieRecomHigh[i] = sorted(enumerate(list(movieRecom[i])), key = lambda x:x[1], reverse=True)[:10]

#将推荐列表存储为列表的形式
RecomList = []
for key,value in movieRecomHigh.items():
    user = userMap[key]
    for movieId, val in value:
        RecomList.append([user, moviesMap[movieId][1]])

Recommend = pd.DataFrame(RecomList, columns=['userId', 'movieId'])
Recommend = pd.merge(Recommend, moviesDF[['movieId', 'title']], on='movieId', how='inner')
Recommend.to_csv("Recommend.csv")