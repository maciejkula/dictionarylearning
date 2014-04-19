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

# Install

Clone the repository and run ``maven install``.

# Dependencies
The package depends on Mahout and Guava IO.



