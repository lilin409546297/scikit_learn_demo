# !/usr/bin/python
# -*- coding: UTF-8 -*-

# k-近邻算法
import pandas
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def k_near():
    """
        2个样本，3个特征
        a(a1,a2,a3),b(b1,b2,b3)
        欧式距离：
             ____________________________________
        p = √(a1 -b1)^2 + (a2-b2)^2 + (a3 - b3)^2
    """
    # 1、原始数据
    # 读取数据
    train_data = pandas.read_csv("k_near/train.csv")
    # print(train_data.head(10))

    # 2、数据处理
    # 数据筛选
    train_data = train_data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 转换时间
    time_value = pandas.to_datetime(train_data["time"], unit="s")
    # 转换成字典
    time_value = pandas.DatetimeIndex(time_value)
    # print(time_value)

    # 构造特征
    data = train_data.copy()
    data["day"] = time_value.day
    data["hour"] = time_value.hour
    data["weekday"] = time_value.weekday
    # print(train_data.head(10))

    # 删除影响特征的数据,axis为1纵向删除
    data = data.drop(["time"], axis=1)

    # 删除小于目标值的数据
    place_count = data.groupby("place_id").count()
    # print(place_count)
    # 过滤数量大于5的地点ID，并且加入列中
    tf = place_count[place_count.x > 5].reset_index()
    # print(tf)
    data = data[data["place_id"].isin(tf.place_id)]

    # 取特征值和目标值
    y = data["place_id"]
    x = data.drop(["place_id", "row_id"], axis=1)

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3、特征工程
    # 特征工程(标准化)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 4、算法
    # 算法计算
    """
        优点：
            简单、易于理解、易于实现、无需估计参数、无需训练
        缺点：
            懒惰算法，对测试样本分类时的计算量大，内存开销大
            必须指定K值，K值选择不当则分类精度不能保证
        问题：
            k值比较小：容易受异常点影响
            k值比较大：容易受K值影响(类别)影响
            性能问题：每一个数据都要循环计算
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print("预测值：", y_predict)

    # 5、评估
    # 评估
    score = knn.score(x_test, y_test)
    print("准确率：", score)

def k_near_test():
    # 1、原始数据
    li = load_iris()
    # print(li.data)
    # print(li.DESCR)
    # 2、处理数据
    data = li.data
    target = li.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
    # 3、特征工程
    std = StandardScaler()
    x_train = std.fit_transform(x_train, y_train)
    x_test = std.transform(x_test)
    # 4、算法
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)
    # 预估
    y_predict = knn.predict(x_test)
    print("预估值：", y_predict)
    # 5、评估
    source = knn.score(x_test, y_test)
    print("准确率：", source)

    """
        交叉验证与网格搜索：
            交叉验证：
                1、将一个训练集分成对等的n份（cv值）
                2、将第一个作为验证集，其他作为训练集，得出准确率
                3、将第二个作为验证集，其他作为训练集，知道第n个为验证集，得出准确率
                4、把得出的n个准确率，求平均值，得出模型平均准确率
            网格搜索：
                1、用于参数的调整（比如，k近邻算法中的n_neighbors值）
                2、通过不同参数传入进行验证（超参数），得出最优的参数值（最优n_neighbors值）
    """
    # 4、算法
    knn_gc = KNeighborsClassifier()
    # 构造值进行搜索
    param= {"n_neighbors": [2, 3, 5]}
    # 网格搜索
    gc = GridSearchCV(knn_gc, param_grid=param,cv=4)
    gc.fit(x_train, y_train)

    # 5、评估
    print("测试集的准确率：", gc.score(x_test, y_test))
    print("交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型：", gc.best_estimator_)
    print("每个超参数每次交叉验证结果：", gc.cv_results_)


def bayes():
    """
        贝叶斯公式：
                     P(W|C)P(C)
            P(C|W) = ——————————
                        P(W)
            说明：P为概率，|在C的前提下W的概率， C分类， W多个条件（特征值）
        文档：
            P(C):每个文档类别的概率（某文档类别数/文档类别总数）
            P(W|C):给定类别特征（被预测文档中出现的词）的概率
        拉普拉斯平滑：
            避免出现次数为0的时候，计算结果直接为0
                      Ni + a
            P(F1|C) = ———————
                      N + am
            说明：a指系数一般为1， m为W(多个条件)的个数，NI为每个条件的个数，N为W（多个条件）的总个数
        优点：
            源于古典数学理论，有稳定的分类效率
            对缺失数据不太敏感，算法比较简单，常用语文本
            分类精确度高，速度快
        缺点：
            使用样本属性独立性假设，与样本属性关联密切。如果训练集准确率不高，会影响结果
    """
    # 1、原始数据
    news = fetch_20newsgroups()

    # 2、处理数据
    x_train, x_test, y_train, y_test= train_test_split(news.data, news.target, test_size=0.25)

    # 3、特征工程
    # 抽取特征数据
    tf = TfidfVectorizer()
    # 训练集中词的重要性统计
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    # 根据训练集转换测试集
    x_test = tf.transform(x_test)

    # 4、算法
    mlt = MultinomialNB()
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测值：", y_predict)

    # 5、评估
    source = mlt.score(x_test, y_test)
    print("准确率：", source)

    # 精准率和召回率
    """
        二分类的算法评价指标（准确率、精准率、召回率、混淆矩阵、AUC）
        数据：
            	预测值 0	预测值 1
        真实值 0	 TN	      FP
        真实值 1	 FN	      TP
        
        精准率（precision）：
                          TP
            precision = ——————   
                        TP + FP
        召回率（recall）：
                       TP
            recall = ———————
                     TP + FN
        模型的稳定性：
                      2TP        2precision * recall
            F1 = ————————————— = ———————————————————
                 2TP + FN + FP    precision + recall
    """
    print("精准率和召回率：\n", classification_report(y_test, y_predict, target_names=news.target_names))

def decision_tree():
    """
        决策树：
            信息熵：
                         n
                H(X) = - ∑ p(x)logp(x)
                        i=1
                说明：log 低数为2，单位比特，H(X)为熵,x为特征具体值，p(x)为该值在x特征值中的概率
            信息增益：
                g(D, A) = H(D) - H(D|A)
        优点：
            简化理解和解释，树木可视化
            需要很少的数据准备，其他技术通常需要数据归一化
        缺点：
            树太过于复杂，过拟合
        改进：
            减枝cart算法(决策树API中已经实现)
        随机森林：
            在当前所有算法中具有极好的准确率
            能够有效的运行在大数据集上
            能够处理具有高维特征的输入样本中，而且不需要降维
            能够评估各个特征在分类问题上的重要性
    """
    # 1、原始数据
    taitan = pandas.read_csv("decision_tree/titanic.csv")
    # 2、数据处理
    x = taitan[["pclass", "age", "sex"]]
    y = taitan["survived"]
    # 缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 3、特征工程
    # 采用DictVectorizer目的是，数据更多是文本类型的，借助dict的方式来处理成0/1的方式
    dict = DictVectorizer(sparse=True)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(x_train)
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print(dict.get_feature_names())
    # 4、算法
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    # 5、评估
    score = tree.score(x_test, y_test)
    print("准确率：", score)

    # 导出决策树图形
    export_graphviz(tree, out_file="decision_tree/tree.dot", feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女', '男'])

    # 随机森林
    # 4、算法
    rf = RandomForestClassifier()
    # 超参数调优
    # 网络搜索与交叉验证
    params = {
        "n_estimators": [120, 200, 300, 500, 800, 1200],
        "max_depth": [5, 8, 15, 25, 30]
    }
    gc = GridSearchCV(rf, param_grid=params, cv=2)
    gc.fit(x_train, y_train)
    # 5、评估
    score = gc.score(x_test, y_test)
    print("准确率：", score)
    print("最佳参数模型：", gc.best_params_)

if __name__ == '__main__':
    # k_near()
    # k_near_test()
    # bayes()
    decision_tree()