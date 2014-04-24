# Dictionary Learning

This package provides an implementation of online dictionary learning for Java, with an emphasis on learning very high-dimensional and very sparse dictionaries. It is based on J. Mairal et al 'Online Dictionary Learning for Sparse Coding' (http://www.di.ens.fr/willow/pdfs/icml09.pdf).  In addition, it implements L2 and L1 regularization of the learned dictionary atoms, leading to sparse dictionaries and easy replacement of uninformative atoms (they are shrunk to zero and replaced by samples from the training data). Because the dictionary is internally implemented as a sparse matrix, the implementations is memory efficient.

# Usage

To train:
```java
int numberOfAtoms = 100;
double l1Penalty = 0.15;
double l2Penalty = 0.01;

Matrix data = readData();

DictionaryLearner dictionaryLearner = new DictionaryLearner(numberOfAtoms, data.columnSize(), new LSMRTransformer());
dictionaryLearner.setL1Penalty(l1Penalty);
dictionaryLearner.setL2Penalty(l2Penalty);
for (Vector row : data) {
    dictionaryLearner.train(row);
}
```

To transform and reconstruct:
```java
Vector dictionaryProjection = dictionaryLearner.transform(datapoint);
Vector reconstructedDatapoint = dictionarLearner.inverseTransform(dictionaryProjection);
```

# Examples

## MovieLens 100K recommendations
This toy example using the MovieLens 100K dataset shows that dictionary learning can be useful for implementing collaborative filtering recommender systems.

One classic way of implementing a recommender system is to use similarity between users to recommend new products: the basic idea is that if user A is very similar to user B (in terms of the products they bought), then any product that A purchased but B did not is a good recommendation for B. 

This is usually implemented by representing users by sparse vectors with positive entries representing products bought, and zeroes representing products which were not bought (in the implicit feedback formulation). We can then compute recommendations for some user A by taking a weighted sum of other users' vectors, where we give higher weight to users more similar to user A. Quantities such as Pearson correlation or cosine distance between the user vectors are often used to derive these weights.

One reason why these weighting schemes may be suboptimal is because the weights are computed independently of each other and are not derived from any optimization scheme. We can envisage two users B and C who are identical to each other and very similar to user A: then both of them will be accorded the same high weight when computing recommendations for A even though (conditional on the other) they do not add any additional information.

We can try to solve this by treating the problem of finding weights as a linear regression problem, where we try to solve for the best (in the squared error sense) linear combination of other users' vectors. This, however, is impractical: it implies repeatedly solving a very large least squares problem (with millions of users and items).

This is where dictionary learning comes in. Since the learned dictionary is a good reduced-dimensionality representation of the space of user vectors, we can compute recommendations as linear combinations of the dictionary atoms instead of raw user vectors. Because the dimensionality of the dictionary is much lower, we can easily compute the optimal weights by projecting the user vectors on the dictionary atoms.

This example uses the implicit feedback form of the MovieLens dataset: rows in the data matrix represent users, columns represent movies. An entry is 1 if a user has rated a given movie, 0 otherwise. The data is split into a training and a test set. The metric used is the average rank of relevant items: an item has rank 0 if it is the most highly recommended item, and 1 if it is the least recommended item. A good recommendation system will make the rank of relevant items (1s in our data matrix) as low as possible.

If we do not perform any training, we obtain a score of 0.5 (as expected for random recommendations). We can then set some parameters and go through the training examples:
```java
DictionaryLearner dictionaryLearner = new DictionaryLearner(256, numMovies, new LSMRTransformer());
dictionaryLearner.setL1Penalty(0.15);
        
Long trainingStartTime = System.currentTimeMillis();
for (Vector row : trainingData) {
    dictionaryLearner.train(row);
    }
System.out.println(String.format("Finished training in %s ms", System.currentTimeMillis() - trainingStartTime));
```

After training, the rank decreases to around 0.04, showing an improvement over the (admittedly not very demanding) random baseline:
```java
double testSetAverageRank = 0.0;
int testSetUsers = 0;
for (Vector row : testData) {
    if (row.getNumNonZeroElements() > 0) {
        testSetUsers++;
        testSetAverageRank += EvaluationUtils.computeAverageNonzeroElementPercentageRank(row, 
                              dictionaryLearner.inverseTransform(dictionaryLearner.transform(row)));
    }
}
testSetAverageRank = testSetAverageRank/testSetUsers;
```

To run the the example, run the following to get the MovieLens 100K dataset from the GroupLens website and execute the training code:
```shell
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip
mv ml-100k/u1.base ./ && mv ml-100k/u1.test ./
mvn exec:java -Dexec.mainClass="com.github.maciejkula.dictionarylearning.MovieLensExample"
```

# Install

Clone the repository and run ``maven install``.

# Dependencies
The package depends on Mahout and Guava IO.



