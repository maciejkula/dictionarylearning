package com.github.maciejkula.dictionarylearning;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

/**
 * Linear algebra routines taking advantage of the fact that
 * the dictionary is a SparseColumnMatrix (and so column operations
 * are fast, and row operations are slow).
 */
public class MathUtils {
	
	/*
	 * Used for computing the left-hand side of the dictionary
	 * projection linear system.
	 * 
	 * Returns the dense result of A'A. 
	 */
    public static Matrix transposedDictionaryTimesDictionary(SparseColumnMatrix dictionary) {
    	Matrix output = new DenseMatrix(dictionary.numCols(), dictionary.numCols());
    	for (int i=0; i < dictionary.numCols(); i++) {
    		for (int j=0; j < dictionary.numCols(); j++) {
    			Vector row = dictionary.viewColumn(i);
    			Vector column = dictionary.viewColumn(j);
    			double value = 0.0;
    			for (Element elem : row.nonZeroes()) {
    				value = value + elem.get() * column.getQuick(elem.index());
    			}
    			output.setQuick(i, j, value);
    		}
    	}
    	return output;
    }
    
    /*
     * Used for computing the right-hand side of the dictionary projection
     * linear system.
     * 
     * Returns the dense result of A'y.
     */
    public static Vector transposedDictionaryTimesDatapoint(SparseColumnMatrix dictionary, Vector datapoint) {
    	Vector output = new DenseVector(dictionary.numCols());
    	for (int i=0; i < dictionary.numCols(); i++) {
    		Vector row = dictionary.viewColumn(i);
    		double value = 0.0;
    		for (Element elem : row.nonZeroes()) {
    			value = value + elem.get() * datapoint.get(elem.index());
    		}
    		output.setQuick(i, value);
    	}
    	return output;
    }
    
    public static Vector inverseTransform(SparseColumnMatrix dictionary, Vector projection) {
    	Vector output = new RandomAccessSparseVector(dictionary.numRows());
    	for (int i=0; i < dictionary.numCols(); i++) {
    		Vector atom = dictionary.viewColumn(i);
    		double projectionWeight = projection.get(i);
    		for (Element elem : atom.nonZeroes()) {
    			output.incrementQuick(elem.index(), elem.get() * projectionWeight);
    		}
    	}
    	return output;
    }

}
