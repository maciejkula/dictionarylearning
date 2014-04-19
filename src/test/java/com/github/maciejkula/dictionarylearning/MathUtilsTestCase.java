package com.github.maciejkula.dictionarylearning;

import java.util.Random;

import junit.framework.TestCase;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;

import com.github.maciejkula.dictionarylearning.MathUtils;

public class MathUtilsTestCase extends TestCase {
	
	private static Vector createRandomVector(int cardinality, double density) {
		Random randomGenerator = new Random();
		Vector vector = new RandomAccessSparseVector(cardinality);
		for (int j=0; j < Math.min(density * cardinality, cardinality); j++) {
			vector.setQuick(randomGenerator.nextInt(cardinality), randomGenerator.nextDouble());
		}
		return vector;
	}
	
	private static SparseColumnMatrix createRandomMatrix(int rows, int cols, double density) {
		SparseColumnMatrix matrix = new SparseColumnMatrix(rows, cols);
		for (int i=0; i < cols; i++) {
			Vector column = matrix.viewColumn(i);
			column.assign(createRandomVector(rows, density));
		}
		return matrix;
	}
	
	private static Boolean assertMatricesEqual(Matrix a, Matrix b) {
		if (a.rowSize() != b.rowSize()) {
			return false;
		}
		if (a.columnSize() != b.columnSize()) {
			return false;
		}
		for (int i=0; i < a.columnSize(); i++) {
			if (!assertVectorsEqual(a.viewColumn(i), b.viewColumn(i))) {
				return false;
			}
		}
		return true;
	}
	
	private static Boolean assertVectorsEqual(Vector a, Vector b) {
		if (a.getDistanceSquared(b) > 0.000000001) {
			return false;
		}
		return true;
	}
	
	public void testTransposedDictionaryTimesDictionary() {
		SparseColumnMatrix matrix = createRandomMatrix(100, 20, 0.2);
		assertTrue(assertMatricesEqual(MathUtils.transposedDictionaryTimesDictionary(matrix),
				matrix.transpose().times(matrix)));
	}
	
	public void testTransposedDictionaryTimesDatapoint() {
		SparseColumnMatrix matrix = createRandomMatrix(100, 20, 0.2);
		Vector vector = createRandomVector(100, 0.3);
		assertTrue(assertVectorsEqual(MathUtils.transposedDictionaryTimesDatapoint(matrix, vector),
				matrix.transpose().times(vector)));
	}
	
	public void testInverseTransform() {
		SparseColumnMatrix matrix = createRandomMatrix(100, 20, 0.2);
		Vector vector = createRandomVector(20, 0.3);
		assertTrue(assertVectorsEqual(MathUtils.inverseTransform(matrix, vector),
				matrix.times(vector)));
	}

}
