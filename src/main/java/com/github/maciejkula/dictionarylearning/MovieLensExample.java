package com.github.maciejkula.dictionarylearning;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

import com.google.common.base.Splitter;

public class MovieLensExample {
    
    private static int numUsers = 943;
    private static int numMovies = 1682;

    private static Rating parseRating(String string) {
        List<String> tokens = Splitter.on('\t').splitToList(string);

        return new Rating(Integer.parseInt(tokens.get(0)),
                Integer.parseInt(tokens.get(1)),
                Double.parseDouble(tokens.get(2)));
    }

    private static List<Rating> readData(String filename) {

        File dataFile = new File(filename);
        List<Rating> data = new ArrayList<Rating>();

        try (BufferedReader reader = new BufferedReader(new FileReader(dataFile))) {  
            String line;
            while ((line = reader.readLine()) != null) {
                data.add(parseRating(line));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return data;
    }
    
    public static class Rating {
        public int userId;
        public int movieId;
        public double rating;

        public Rating(int userId, int movieId, double rating) {
            this.userId = userId - 1;
            this.movieId = movieId - 1;
            this.rating = rating;
        }  
    }
    
    /*
     * Toy implementation of a basic neighbourhood-based collaborative filtering recommender,
     * where recommendations for user X are a sum over all user vectors weighted by the cosine
     * of the angle between the user vectors and X's user vector.
     * 
     */
    public static class NeighbourhoodCF {
    	
    	private final Matrix transposedDataMatrix;
    	private final Matrix similarityMatrix;
    	
    	public NeighbourhoodCF(Matrix dataMatrix) {
    		this.transposedDataMatrix = new DenseMatrix(dataMatrix.numCols(), dataMatrix.numRows());
    		this.transposedDataMatrix.assign(dataMatrix.transpose());
    		this.similarityMatrix = this.computeSimilarityMatrix(dataMatrix);
    	}
    	
    	public Vector getRecommendations(int userId) {
    		Vector similarities = this.similarityMatrix.viewRow(userId);
    		return new RandomAccessSparseVector(this.transposedDataMatrix.times(similarities));
    	}
    	
    	/*
    	 * Calculate the cosine of the angle between the two vectors.
    	 */
    	public double computeSimilarity(Vector a, Vector b, double aNorm, double bNorm) {
    		return a.dot(b) / (aNorm * bNorm);
    	}
    	
    	public Matrix computeSimilarityMatrix(Matrix dataMatrix) {
    		
    		DenseMatrix similarityMatrix = new DenseMatrix(dataMatrix.numRows(), dataMatrix.numRows());
    		double[] norms = new double[dataMatrix.numRows()];
    		for (int i=0; i < dataMatrix.numRows(); i++) {
    			norms[i] = dataMatrix.viewRow(i).norm(2);
    		}
    		
    		for (int i=0; i < dataMatrix.numRows(); i++) {
    			for (int j=i; j < dataMatrix.numRows(); j++) {
    				double similarity = 0.0;
    				if (i != j) {
    					similarity = computeSimilarity(dataMatrix.viewRow(i), dataMatrix.viewRow(j), norms[i], norms[j]);
    				}
    				similarityMatrix.setQuick(i, j, similarity);
    				similarityMatrix.setQuick(j, i, similarity);
    			}
    		}
    		return similarityMatrix;
    	}
    }
    
    private static Matrix createRatingMatrix(String filename) {
           Matrix dataMatrix = new SparseRowMatrix(numUsers, numMovies);
           for (Rating rating : readData(filename)) {
               dataMatrix.setQuick(rating.userId, rating.movieId, 1.0);
           }        
           return dataMatrix; 
    }
    
    public static void main(String[] args) {
        
        Path currentRelativePath = Paths.get("");
        String path = currentRelativePath.toAbsolutePath().toString();
        System.out.println("Looking for u1.base and u1.test in " + path);
        
        Matrix trainingData = createRatingMatrix("u1.base");
        Matrix testData = createRatingMatrix("u1.test");
        
        System.out.println("Finished loading data. Starting training.");
        
        DictionaryLearner dictionaryLearner = new DictionaryLearner(256, numMovies, new LSMRTransformer());
        dictionaryLearner.setL1Penalty(0.15);
        
        Long trainingStartTime = System.currentTimeMillis();
        for (Vector row : trainingData) {
            dictionaryLearner.train(row);
        }
        System.out.println(String.format("Finished training in %s ms", System.currentTimeMillis() - trainingStartTime));
        
        double trainingSetAverageRank = 0.0;
        for (Vector row : trainingData) {
            trainingSetAverageRank += EvaluationUtils.computeAverageNonzeroElementPercentageRank(row, 
            		dictionaryLearner.inverseTransform(dictionaryLearner.transform(row)));
        }
        trainingSetAverageRank = trainingSetAverageRank/numUsers;
        
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
        
        System.out.println(String.format("Dictionary learning average rank in training set: %s,  average rank in test set %s.", 
        		trainingSetAverageRank, testSetAverageRank));
        
        // Do the same for neighbourhood-based recommendations
		System.out.println("Computing similarity matrix");
        NeighbourhoodCF nCF = new NeighbourhoodCF(trainingData);
		System.out.println("Similarity matrix computed");
        
        trainingSetAverageRank = 0.0;
        for (int i=0; i < trainingData.numRows(); i++) {
            trainingSetAverageRank += EvaluationUtils.computeAverageNonzeroElementPercentageRank(trainingData.viewRow(i), 
            		nCF.getRecommendations(i));
        }
        trainingSetAverageRank = trainingSetAverageRank/numUsers;
        
        testSetAverageRank = 0.0;
        testSetUsers = 0;
        for (int i=0; i < testData.numRows(); i++) {
        	Vector row = testData.viewRow(i);
            if (row.getNumNonZeroElements() > 0) {
            	testSetUsers++;
            	double score = EvaluationUtils.computeAverageNonzeroElementPercentageRank(row, 
            			nCF.getRecommendations(i));
            	testSetAverageRank += score;
            }
        }
        testSetAverageRank = testSetAverageRank/testSetUsers;
        
        System.out.println(String.format("Neighbourhood CF average rank in training set: %s,  average rank in test set %s.", 
        		trainingSetAverageRank, testSetAverageRank));
    }

}
