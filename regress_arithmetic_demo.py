# !/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def regression():
    """
        属性的线性组合：
            f(x) = w1x1 + w2x2 + w3x3 + ... + wnxn + b
            w：权重， b偏置项， x：特征数据
            b：单个特征是更加通用
        线性回归：
            通过一个或者多个自变量与因变量之间进行建模的回归分析
            其中可以为一个或者多个自变量之间的线性组合（线性回归的一种）
            一元线性回归：
                涉及的变量只有一个
            多元线性回归：
                涉及变量为两个或者两个以上
            通用公式：
                h(w) = w0 + w1x1 + w2x2 + ... + wnxn
                w,x为矩阵w0为b
        矩阵：
            必须是二维
            乘法公式：
                （m行， l列）* （l行， n列） = （m行， n列）
                例如：
                    [[1,2,3,4]] * [[5],[6],[7],[8]] = 5 * 1 + 6 * 2 + 7 * 3 + 8 * 4
        损失函数(最小二乘法)(误差的平方和)：
            j(θ) = (hw(x1) - y1)^2 + (hw(x2) - y2)^2 + ... + (hw(xn) - yn)^2
                   n
                 = ∑(hw(xi) - yi)^2
                  i=1
            yi：训练样本的真实值， hw(xi)：第i个训练样本的特征、组合预测值
        权重：
            正规方程：
                W = (XtX)^(-1)XtY
                X：特征值矩阵， Y：目标值矩阵 Xt：转置特征值（行列替换）
                特征比较复杂时，不一定能得出结果
            梯度下降：
                例子(单变量)：
                                δcost(w0 + w1x1)    ||
                    w1 = -w1 - α————————————————    ||
                                      δw1           || (下降)
                                δcost(w0 + w1x1)    ||
                    w0 = -w0 - α————————————————    ||
                                      δw1           \/
                    α：学习速率，需要手动指定
                    δcost(w0 + w1x1)
                    ———————————————— 表示方向
                          δw1
        回归性能评估：
                   1  m      _
            MSE = ——— ∑(yi - y)^2
                   m i=1
                      _
            yi：预测值 y：真实值
            一定要标准化之前的值
        对比：
            梯度下降：
                1、需要选择学习率α
                2、需要多次迭代
                3、当特征数量n很大时，也比较适用
                4、适用于各种类型的模型
            正规方程：
                1、不需要选择学习率α
                2、一次运算得出
                3、需要计算(XtX)^(-1), 如果特征数量n很大时，时间复杂度很高，通常n<100000,可以接受
                4、只能用于线性模型，不适合逻辑回归模型等其他模型
        岭回归：
            1、因为线性回归（LinearRegression）容易出现过拟合的情况，所有需要正则化
            2、正则化的目的，就是将高幂（x^n,n很大），的权重降到接近于0
            3、岭回归为带有正则化的线性回归
            4、回归得到的系数更加符合实际，更加可靠，更存在病态数据偏多的研究中存在较大价值
        Ridge:
            1、具有l2正则化的线性最小二乘法
            2、alpha(λ):正则化力度
            3、coef_:回归系数
    """
    # 1、获取数据
    lb = load_boston()

    # 2、处理数据
    # 分隔数据
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 3、特征工程
    # 数据标准化(目的，特征值差异过大，按比例缩小)
    # 目标值也要进行标准化（目的，特征值标准化后，特征值值过大在回归算法中，得出的权重值差异过大）
    # 两次标准化实例的目的，就是不同数据之间的实例化不一样
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    # 目标值也要转成2维数组(-1,不知道样本数)
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))
    # print(x_train, y_train)


    # 4、线性回归正规算法
    """
        1、通过结果可以看出真实值和预测值的差距还是很大的。
        2、这是直接通过线性回归的正确公式来算出权重值的结果。
        3、为了更好的减少误差，所以采用梯度下降的方式，来重新计算权重值
    """
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predict_lr = lr.predict(x_test)
    # 注意这里的预测值是标准化过后的数据，需要转回来
    # print("预测值：", std_y.inverse_transform(y_predict_lr).reshape(1, -1))
    # print("真实值：", std_y.inverse_transform(y_test).reshape(1, -1))
    print("权重值：", lr.coef_)

    # 5、回归评估
    print("正规方程均方误差：", mean_squared_error(std_y.inverse_transform(y_test).reshape(1, -1), std_y.inverse_transform(y_predict_lr).reshape(1, -1)))

    # 4、线性回归梯度下降算法
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    y_predict_sgd = sgd.predict(x_test)
    # 注意这里的预测值是标准化过后的数据，需要转回来
    # print("预测值：", std_y.inverse_transform(y_predict_sgd).reshape(1, -1))
    # print("真实值：", std_y.inverse_transform(y_test).reshape(1, -1))
    print("权重值：", sgd.coef_)

    # 5、回归性能评估
    print("梯度下降均方误差：", mean_squared_error(std_y.inverse_transform(y_test).reshape(1, -1), std_y.inverse_transform(y_predict_sgd).reshape(1, -1)))

    # 4、线性回归正则化算法（岭回归）
    # alpha为超参数，可以通过网格搜索和交叉验证，来确认alpha的值
    # alpha范围（0~1， 1~10）
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    y_predict_rd = rd.predict(x_test)
    # 注意这里的预测值是标准化过后的数据，需要转回来
    # print("预测值：", std_y.inverse_transform(y_predict_rd).reshape(1, -1))
    # print("真实值：", std_y.inverse_transform(y_test).reshape(1, -1))
    print("权重值：", sgd.coef_)

    # 5、回归性能评估
    print("正则化均方误差：", mean_squared_error(std_y.inverse_transform(y_test).reshape(1, -1), std_y.inverse_transform(y_predict_sgd).reshape(1, -1)))


"""
    欠拟合：
        原因：
            学习到的特征太少
        解决办法：
            增加数据的特征数量
    过拟合（训练集和测试集表现不好）：
        原因：
            原始特征数量过多，存在一些嘈杂的特征，模型过于复杂是因为模型尝试去兼顾各个测试数据点
        解决办法：
            进行特征选择，消除一些关联性不大的特征（不好做）
            交叉验证（让所有数据进行训练）
            正则化
"""

if __name__ == '__main__':
    regression()