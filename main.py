# this is practice for working with deep learning in python using keras and python
#importing tensorflow
import tensorflow as tf

#determines what data set we are using and converts it from integers to floating point
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#determines the sequential model of layers that are applied to the deep learnign algorithm
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
#this model returns vecotr of log-odd scones for each class
#this is almost meaningless for us at the moment
predictions = model(x_train[:1]).numpy()

#############################################################################################
#this line covertes the preditions above to probabilities for each class
#Note it is possible to integrate this into the last layer of the network however unadvised
#makes it impossible to provide an exact numerically stable loss calcuation for all models using softmax output
#############################################################################################
tf.nn.softmax(predictions).numpy()

#losses.SparseCategoricalCrossentropy is a loss that takes a vector of logits and returns a loss for each example when set to true
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

#adam is the name of te optimizer  learning rate 0.001, beta_1 = 0.9, beta_2 =0.999, epsilon = 1e07, amsgrad=false
#the rest of the variables in the compile should be self explanatory
model.compile(optimizer= 'adam', loss = loss_fn, metrics = ['accuracy'])
#adjusts model parameteres to minimize the loss
model.fit(x_train, y_train, epochs = 5)

# checks model performace on a validation -set, test set
model.evaluate(x_test, y_test, verbose=2)

#at this point the model is 98 % trained on the current dataset
#to return a probability swrap the trained model and attach the softmax like the code below excluding line 47
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()
                                         ])
print(probability_model(x_test[:5]))


