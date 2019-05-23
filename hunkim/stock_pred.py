'''
The original script shows how to predict the next day's closing stock prices using a basic RNN
https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py

I am making a change with two points.
 [1] adding various inputs
    ex) stock price of other nations, newspaper, etc.
# Open, High, Low, Volume, Close
I will change the shape of data
[Open, High, Low, Volume, Close] -> [[nation, date, Open, High, Low, Volume, Close]]
 [2] predicting multiple values
    ex) long-term movement(60-days, 15-days, the next day)

At first, let's understand the original code and prior arts completely
'''


# https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-12-5-rnn_stock_prediction.py
def hunkim_stock():
    import tensorflow as tf
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import os

    tf.set_random_seed(777)  # reproducibility
    np.set_printoptions(precision=2)

    if "DISPLAY" not in os.environ:
        # remove Travis CI Error
        matplotlib.use('Agg')


    def MinMaxScaler(data):
        '''
        Min Max Normalization
        Parameters
        ----------
        data : numpy.ndarray
            input data to be normalized
            shape: [Batch size, dimension]
        Returns
        ----------
        data : numpy.ndarry
            normalized data
            shape: [Batch size, dimension]
        References
        ----------
        .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        원래는 normalized data만 반환하였으나, 데이터의 복구를 위해 min, max도 반환
        '''
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return [numerator / (denominator + 1e-7), np.min(data, 0), np.max(data, 0)]


    # train Parameters
    seq_length = 7
    data_dim = 5
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.01
    iterations = 500

    # Open, High, Low, Volume, Close
    xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
    xy_rev = xy[::-1]  # reverse order (chronically ordered), 날짜 오름차순으로.
    '''
    print('xy: ', xy[-3:])
    xy:  [[  566.89   567.     556.93 10800.     556.97]
     [  561.2    566.43   558.67 41200.     559.99]
     [  568.     568.     552.92 13100.     558.46]]
    print('xy_rev: ', xy_rev[:3])
    xy:  [[  568.     568.     552.92 13100.     558.46]
     [  561.2    566.43   558.67 41200.     559.99]
     [  566.89   567.     556.93 10800.     556.97]]
    '''

    # split data to train_set/test_set and Scaling
    train_size = int(len(xy_rev) * 0.7)
    train_set = xy_rev[0:train_size]
    test_set = xy_rev[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence
    [train_set, min, max] = MinMaxScaler(train_set)
    [test_set, min, max] = MinMaxScaler(test_set)
    '''
    print('train_set: ', train_set[:3])
    print('min: ', min)     # 컬럼별로 min-max 연산은 따로따로 한 것을 알 수 있음.!!!
    train_set:  [[0.25 0.25 0.23 0.   0.23]
                 [0.23 0.24 0.25 0.   0.24]
                 [0.25 0.24 0.25 0.   0.23]]
    min:  [ 494.65  495.98  487.56 7900.    492.55]
    '''


    # build datasets. Create batch for 7-days.
    def build_dataset(time_series, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length):
            _x = time_series[i:i + seq_length, :]
            _y = time_series[i + seq_length, [-1]]  # the next day's closing stock prices
            # print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)
        return np.array(dataX), np.array(dataY)


    trainX, trainY = build_dataset(train_set, seq_length)
    testX, testY = build_dataset(test_set, seq_length)
    '''
    print('trainX: ', trainX[:4])
    print('trainY: ', trainY[:3])
    trainX:  [[[2.53e-01 2.45e-01 2.34e-01 4.66e-04 2.32e-01]
      [2.30e-01 2.40e-01 2.55e-01 2.98e-03 2.37e-01]
      [2.49e-01 2.42e-01 2.48e-01 2.60e-04 2.27e-01]
      [2.21e-01 2.47e-01 2.55e-01 0.00e+00 2.63e-01]
      [3.63e-01 3.70e-01 2.67e-01 1.25e-02 2.62e-01]
      [2.59e-01 3.11e-01 2.74e-01 4.56e-01 2.72e-01]
      [2.76e-01 2.78e-01 1.98e-01 5.70e-01 1.78e-01]]
    
     [[2.30e-01 2.40e-01 2.55e-01 2.98e-03 2.37e-01]
      [2.49e-01 2.42e-01 2.48e-01 2.60e-04 2.27e-01]
      [2.21e-01 2.47e-01 2.55e-01 0.00e+00 2.63e-01]
      [3.63e-01 3.70e-01 2.67e-01 1.25e-02 2.62e-01]
      [2.59e-01 3.11e-01 2.74e-01 4.56e-01 2.72e-01]
      [2.76e-01 2.78e-01 1.98e-01 5.70e-01 1.78e-01]
      [1.59e-01 1.79e-01 1.42e-01 3.94e-01 1.61e-01]]
    
     [[2.49e-01 2.42e-01 2.48e-01 2.60e-04 2.27e-01]
      [2.21e-01 2.47e-01 2.55e-01 0.00e+00 2.63e-01]
      [3.63e-01 3.70e-01 2.67e-01 1.25e-02 2.62e-01]
      [2.59e-01 3.11e-01 2.74e-01 4.56e-01 2.72e-01]
      [2.76e-01 2.78e-01 1.98e-01 5.70e-01 1.78e-01]
      [1.59e-01 1.79e-01 1.42e-01 3.94e-01 1.61e-01]
      [1.65e-01 2.01e-01 1.93e-01 2.82e-01 2.20e-01]]
    
     [[2.21e-01 2.47e-01 2.55e-01 0.00e+00 2.63e-01]
      [3.63e-01 3.70e-01 2.67e-01 1.25e-02 2.62e-01]
      [2.59e-01 3.11e-01 2.74e-01 4.56e-01 2.72e-01]
      [2.76e-01 2.78e-01 1.98e-01 5.70e-01 1.78e-01]
      [1.59e-01 1.79e-01 1.42e-01 3.94e-01 1.61e-01]
      [1.65e-01 2.01e-01 1.93e-01 2.82e-01 2.20e-01]
      [2.24e-01 2.36e-01 2.34e-01 2.98e-01 2.52e-01]]]
    trainY:  [[0.16]
     [0.22]
     [0.25]]
    '''

    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

    # cost/loss
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                                    X: trainX, Y: trainY})
            if i % 100 ==0:
                print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={
                        targets: testY, predictions: test_predict})
        print("RMSE: {}".format(rmse_val))

        # Plot predictions
        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
    # plt.show()
    plt.savefig('Stock_price.png')

# Time Series Prediction Using LSTM Deep Neural Networks(Jakob Aungiers, 1st September 2018)
# https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

# Stock Prediction using yahoo stock
# https://github.com/tencia/stocks_rnn
# https://github.com/tencia/stocks_rnn/blob/master/train_stock_lstm.py

# search results in github
# https://github.com/search?l=Python&q=stock+prediction&type=Repositories

''' 
1. 각국의 지수는 서로 영향을 줄 것
2. 장기 추세와 단기 추세의 예측은 서로 다른 관점이 작용할 것.
3. 종가 예측 못지 않게, 매수 포인트와 매도 포인트를 예측해주는 것이 주효
4. 주가 이외의 입력 다양화 필요. 
''' # planning