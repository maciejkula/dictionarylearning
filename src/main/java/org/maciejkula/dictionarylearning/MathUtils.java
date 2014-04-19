package org.maciejkula.dictionarylearning;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

public class MathUtils {
	
    public static Matrix squareMatrix(Matrix matrix) {
    	Matrix output = new DenseMatrix(matrix.numCols(), matrix.numCols());
    	for (int i=0; i < matrix.numCols(); i++) {
    		for (int j=0; j < matrix.numCols(); j++) {
    			Vector row = matrix.viewColumn(i);
    			Vector column = matrix.viewColumn(j);
    			double value = 0.0;
    			for (Element elem : row.nonZeroes()) {
    				value = value + elem.get() * column.getQuick(elem.index());
    			}
    			output.setQuick(i, j, value);
    		}
    	}
    	return output;
    }
    
    public static Matrix transposedMatrixVectorMultiplication(Matrix matrix, Vector vector) {
    	Matrix output = new DenseMatrix(matrix.numCols(), 1);
    	for (int i=0; i < matrix.numCols(); i++) {
    		Vector row = matrix.viewColumn(i);
    		double value = 0.0;
    		for (Element elem : row.nonZeroes()) {
    			value = value + elem.get() * vector.get(elem.index());
    		}
    		output.setQuick(i, 0, value);
    	}
    	return output;
    }
    
    public static Vector reproject(Matrix matrix, Vector vector) {
    	Vector output = new RandomAccessSparseVector(matrix.numRows());
    	for (int i=0; i < matrix.numCols(); i++) {
    		Vector atom = matrix.viewColumn(i);
    		double projectionWeight = vector.get(i);
    		for (Element elem : atom.nonZeroes()) {
    			output.incrementQuick(elem.index(), elem.get() * projectionWeight);
    		}
    	}
    	return output;
    }

}
