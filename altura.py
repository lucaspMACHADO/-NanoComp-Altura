from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.initializers import RandomUniform
import numpy

numpy.random.seed(13)
RandomUniform(seed=13) #peguei da rede do ufop. Procurar saber como funciona


model = Sequential()

#model.add(Dense(12, imput_dim=6, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))

#Estava fazendo da forma acima e nao estava dando muito certo. Copiei do ufop a forma abaixo:

model.add(Dense(6, input_dim=6))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(11))
model.add(Activation('relu'))
model.add(Dropout(0.01))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


dataset = numpy.loadtxt('dados_altura_totais.txt', delimiter=' ') #Esse arquivo esta sem os ultimos 7 dados, que foram passados para o arquivo "test"

for i in range(6):
    X = dataset[0:i*10, 0:6]
    X = numpy.concatenate((X, dataset[(i+1)*10:60, 0:6]), axis=0)

    Y = dataset[0:i*10, 6]
    Y = numpy.concatenate((Y, dataset[(i+1)*10:60, 6]), axis=0)

    model.fit(X, Y, epochs=100, batch_size=1, verbose=0)

    Xt = dataset[i*10:(i+1)*10, 0:6]
    Yt = dataset[i*10:(i+1)*10, 6]

    res = model.evaluate(Xt, Yt, batch_size=1, verbose=1)
    print('\n', res, "\n-------------------------------------------------\n")



datatest = numpy.loadtxt('test', delimiter=' ')

Xtest = datatest[:, 0:6]

Ytest = model.predict(Xtest, batch_size=1, verbose=1)

print('\n\nResultado:\n', Ytest)

backend.clear_session()