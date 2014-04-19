package com.github.maciejkula.dictionarylearning;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.net.URL;
import java.util.ArrayList;

import junit.framework.TestCase;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.utils.vectors.csv.CSVVectorIterator;

import com.github.maciejkula.dictionarylearning.DictionaryLearner;
import com.github.maciejkula.dictionarylearning.LSMRTransformer;
import com.google.common.base.Charsets;
import com.google.common.io.Resources;

public class DictionaryLearningTestCase extends TestCase {

	private static Matrix readData() {
		URL url = Resources.getResource("data.csv");
		String text = "";
		try {
			text = Resources.toString(url, Charsets.UTF_8);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		CSVVectorIterator csv = new CSVVectorIterator(new StringReader(text));
		ArrayList<Vector> vectors = new ArrayList<Vector>();
		while (csv.hasNext()) {
			vectors.add(csv.next());
		}
		return new SparseRowMatrix(vectors.size(), vectors.get(0).size(), vectors.toArray(new Vector[3]));
	}

	private static Matrix sparsifyData(Matrix denseData, int cardinality) {
		ArrayList<Vector> vectors = new ArrayList<Vector>();
		for (Vector denseVector : denseData) {
			Vector sparsefiedVector = new RandomAccessSparseVector(cardinality);
			for (Element elem : denseVector.nonZeroes()) {
				sparsefiedVector.set(elem.index(), elem.get());
			}
			vectors.add(sparsefiedVector);
		}
		return new SparseRowMatrix(vectors.size(), cardinality, vectors.toArray(new Vector[3]));
	}

	public void testAccuracy() {
		Matrix matrix = sparsifyData(readData(), 1048576);
		int numAtoms = 20;
		DictionaryLearner dictionaryLearner = new DictionaryLearner(numAtoms, matrix.columnSize(), new LSMRTransformer());
		dictionaryLearner.setL1Penalty(0.15);
		dictionaryLearner.setL2Penalty(0.01);
		for (Vector row : matrix) {
			dictionaryLearner.train(row);
		}

		System.out.println("Printing the dictionary");
		for (int i = 0; i < numAtoms; i++) {
			System.out.println(dictionaryLearner.getDictionary().viewColumn(i));
		}
		System.out.println("Finished printing the dictionary");

		double squareError = 0.0;
		for (Vector datapoint : matrix) {
			squareError = squareError + datapoint.getDistanceSquared(dictionaryLearner.inverseTransform(dictionaryLearner.transform(datapoint)));
		}
		System.out.println(String.format("Error: %s", squareError));
		assertTrue(squareError < 0.000001);
	}

	public void testSerialization() {

		Matrix data = readData();        
		DictionaryLearner dictionaryLearner = new DictionaryLearner(10, data.columnSize(), new LSMRTransformer());
		dictionaryLearner.setL1Penalty(0.15);
		for (Vector row : data) {
			dictionaryLearner.train(row);
		}

		ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
		DataOutputStream outputStream = new DataOutputStream(byteStream);
		try {
			dictionaryLearner.write(outputStream);
		} catch (IOException e) {
			e.printStackTrace();
		}
		DictionaryLearner deserializedLearner = new DictionaryLearner(new LSMRTransformer());
		DataInputStream input = new DataInputStream(new ByteArrayInputStream(byteStream.toByteArray()));
		try {
			deserializedLearner.readFields(input);
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(dictionaryLearner.equals(deserializedLearner));
	}
}
