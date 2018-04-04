# Positive-Negative-Review-Classifier
I always wanted to classify text. Recently after completing a Course on **Neural Networks** I became so proficient that I apply them to almost **Everything!!**.

Currently the above code is applying a 5 Layer Neural Network with ReLu Activations. I am using Keras for applying the Neural Networks and Numpy to convert the text data into a suitable format. 

First about preprocessing.

First i will read all the files and make a dictionary of all unique words. Next for every word I will specify an Index. Now I will take a sentence say 'How are you'. Now I will create a Array whose length is equal to number of unique words and fill it with zeros. Now for every word in the text sentence I will mark it as 1. Now this Array will have Ones at the Places that represent this word. And we repeat this process for all Words.

Now split the data into Test, Train and Validation Sets. Now we apply a Keras NN.
