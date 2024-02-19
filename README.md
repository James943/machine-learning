I used machine learning algorithms to predict the problems of orthopaedic patients. My system helps the with patient’s diagnostics of spinal conditions, solving an important real-world problem.

The dataset I am using has 310 instances and 6 attributes, and classifies orthopaedic patients with one of 3 diagnostics, which is either normal (NO), disk hernia (DH), or spondylolisthesis (SL). Each patient is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine. These 6 attributes, all of which are important in identifying the class are (included in the same order as in the dataset): pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius, and grade of spondylolisthesis. Out of the 310 entries, 100 patients are categorised are normal, 60 are categorised are disk hernia, and 150 are categorised as spondylolisthesis.


When testing out different classifiers to use on my intelligent system, firstly I tried the K-nearest neighbour classifier with the baseline value of K=3. I then modified this algorithm by altering K to the found optimum value of K=7. Also I implemented scaling to the dataset with the code:
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
This further improved the accuracy of the results as well as the precision, making me believe that this is the optimal method for the dataset.

Next, I used a multi-Layer Perceptron as a classifier to test on the dataset. This proved to give more inconsistent and poorer results than my developed K nearest neighbour algorithm.
![image](https://github.com/James943/machine-learning/assets/80528511/31f0689f-cc97-40b6-9a4d-86fca6346bc9)
My developed K Nearest Neighbour algorithm gave the highest and most consistent scores for accuracy and accuracy when taking the mean of a 5 cross fold validation.
![image](https://github.com/James943/machine-learning/assets/80528511/f08782df-ea50-4b29-bc76-4b45c9cdf618)
The precision of my K nearest neighbour algorithm again proved to be superior over the baseline K nearest neighbour classifier as well as the inconsistent multi-layered perceptron.
![image](https://github.com/James943/machine-learning/assets/80528511/1b25a4ea-b3c7-4b17-92f0-db31e0c29bae)


Overall after adjusting and finding the optimal method to use on my dataset, which is the K nearest neighbour algorithm, I was able to to predict, with high accuracy, the problems of orthopaedic patients from their attributes.
