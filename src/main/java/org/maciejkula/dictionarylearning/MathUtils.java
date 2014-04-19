package org.maciejkula.dictionarylearning;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

public class MathUtils {
	
    public static Matrix transposedDictionaryTimesDIctionary(Matrix dictionary) {
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
    
    public static Matrix transposedDictionaryTimesDatapoint(Matrix dictionary, Vector datapoint) {
    	Matrix output = new DenseMatrix(dictionary.numCols(), 1);
    	for (int i=0; i < dictionary.numCols(); i++) {
    		Vector row = dictionary.viewColumn(i);
    		double value = 0.0;
    		for (Element elem : row.nonZeroes()) {
    			value = value + elem.get() * datapoint.get(elem.index());
    		}
    		output.setQuick(i, 0, value);
    	}
    	return output;
    }
    
    public static Vector inverseTransform(Matrix dictionary, Vector projection) {
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
