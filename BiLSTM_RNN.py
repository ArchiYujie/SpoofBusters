from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
np.random.seed(7)

def read_data_small():
    X_train = pd.read_csv("/X_train_small.csv")
    X_test = pd.read_csv("/X_test_small.csv")
    y_train = np.asarray(pd.read_csv("/y_train_small.csv", header=None)[0])
    return X_train, X_test, y_train


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate


### code classifier here ###
def format_data(df):
    # append numberical columns
    rst = df.loc[:, ["price", "volume", "bestBid", "bestAsk", 'bestBidVolume',
                     'bestAskVolume', 'lv2Bid', 'lv2BidVolume', 'lv2Ask',
                     'lv2AskVolume', 'lv3Bid', 'lv3BidVolume', 'lv3Ask',
                     'lv3AskVolume']]

    # encode the binaries
    rst["isBid"] = df.isBid * 1
    rst["isBuyer"] = df.isBuyer * 1
    rst["isAggressor"] = df.isAggressor * 1
    rst["type"] = (df.type == "ORDER") * 1
    rst["source"] = (df.source == "USER") * 1

    # parse the order id data
    rst["orderId"] = df.orderId.str.split('-').str[-1]
    rst["tradeId"] = df.tradeId.str.split('-').str[-1]
    rst["bidOrderId"] = df.bidOrderId.str.split('-').str[-1]
    rst["askOrderId"] = df.askOrderId.str.split('-').str[-1]

    # encode the multiple lable data
    tmp_operation = pd.DataFrame(pd.get_dummies(df.operation), columns=df.operation.unique()[:-1])
    rst = pd.concat([rst, tmp_operation], axis=1)
    tmp_endUserRef = pd.DataFrame(pd.get_dummies(df.endUserRef), columns=df.endUserRef.unique()[:-1])
    rst = pd.concat([rst, tmp_endUserRef], axis=1)

    # also feel free to add more columns inferred from data
    # smartly engineered features can be very useful to improve the classification resutls
    rst["timeSinceLastTrade"] = X_train[["timestamp", "endUserRef"]].groupby("endUserRef").diff()

    return rst

X_train, X_test, y_train = read_data_small()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
X_clean_train = format_data(X_train)
X_clean_test = format_data(X_test)
X_clean_train = X_clean_train.fillna(-1)
X_clean_test = X_clean_test.fillna(-1)
X_train_clean_scaled = scale(X_clean_train)
X_test_clean_scaled = scale(X_clean_test)

# truncate and pad input sequences
# create the model
model = Sequential()
model.add(Embedding(input_dim=497, output_dim=32))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_clean_scaled, y_train, epochs=1, batch_size=256)
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_test_clean_scaled, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))