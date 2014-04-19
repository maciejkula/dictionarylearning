package org.maciejkula.dictionarylearning;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

public class DictionaryLearner implements Writable {

	private double learningRate = 0.01;
	private double l1Penalty = 0.0;
	private double l2Penalty = 0.0;

	private int numberOfAtoms;
	private int numberOfFeatures;

	private final Transformer transformer;
	private SparseColumnMatrix dictionaryMatrix;

	public DictionaryLearner(int numberOfAtoms, int numberOfFeatures, Transformer transformer) {
		this.numberOfAtoms = numberOfAtoms;
		this.numberOfFeatures = numberOfFeatures;

		this.transformer = transformer;
		this.dictionaryMatrix = this.createEmptyDictionaryMatrix();
	}

	public DictionaryLearner(Transformer transformer) {
		this.transformer = transformer;
	}

	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}

	public double getLearningRate() {
		return this.learningRate;
	}

	public void setL1Penalty(double norm) {
		this.l1Penalty = norm;
	}

	public double getL1Penalty() {
		return this.l1Penalty;
	}

	public void setL2Penalty(double norm) {
		this.l2Penalty = norm;
	}

	public double getL2Penalty() {
		return this.l2Penalty;
	}

	public Matrix getDictionary() {
		return this.dictionaryMatrix;
	}

	/*
	 * Project the datapoint on the dictionary.
	 */
	public Vector transform(Vector datapoint) {
		return this.transformer.transform(datapoint, this.dictionaryMatrix);
	}

	/*
	 * Reconstruct a datapoint from its projection on the dictionary.
	 */
	public Vector inverseTransform(Vector datapoint) {
		return this.transformer.inverseTransform(datapoint, this.dictionaryMatrix);
	}

	/*
	 * Perform an online update of the dictionary using a datapoint.
	 * 
	 * Returns the datapoint's projection on the dictionary atoms.
	 */
	public Vector train(Vector datapoint) {
		this.initializeAtoms(datapoint);		
		Vector projection = this.transformer.transform(datapoint, this.dictionaryMatrix);

		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector atom = this.dictionaryMatrix.viewColumn(i);
			double projectionWeight = projection.get(i);
			Vector difference = atom.minus(datapoint);
			for (Element elem : difference.nonZeroes()) {
				atom.incrementQuick(elem.index(), - this.learningRate * projectionWeight * elem.get());
			}
		}
		this.regularize();
		
		return projection;
	}

	private SparseColumnMatrix createEmptyDictionaryMatrix() {
		return new SparseColumnMatrix(this.numberOfFeatures, this.numberOfAtoms);
	}

	/*
	 * Apply L2 and L1 regularization to dictionary items.
	 */
	private void regularize() {
		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector atom = this.dictionaryMatrix.viewColumn(i);
			List<Integer> indicesToRemove = new ArrayList<Integer>();
			for (Element elem : atom.nonZeroes()) {
				double regularizedValue = elem.get() - (this.learningRate 
						* (elem.get() 
								* this.l2Penalty
								+ this.l1Penalty * Math.signum(elem.get())));
				if (Math.abs(regularizedValue) < this.l1Penalty) {
					indicesToRemove.add(elem.index());
				} else {
					atom.setQuick(elem.index(), regularizedValue);
				}
			}
			for (int indexToRemove : indicesToRemove) {
				atom.setQuick(indexToRemove, 0.0);
			}
		}
	}

	/*
	 * Initialize atoms: atoms with zero nonzero entries (whether
	 * not initialized or shrunk to zero) are replaced with datapoints.
	 */
	private void initializeAtoms(Vector datapoint) {
		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector column =  this.dictionaryMatrix.viewColumn(i);
			if (column.getNumNonZeroElements() == 0) {
				column.assign(datapoint);
				break;
			}
		}
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.setLearningRate(input.readDouble());
		this.setL1Penalty(input.readDouble());
		this.setL2Penalty(input.readDouble());
		this.numberOfAtoms = input.readInt();
		this.numberOfFeatures = input.readInt();
		this.transformer.readFields(input);
		this.dictionaryMatrix = this.createEmptyDictionaryMatrix();
		this.dictionaryMatrix.assign(MatrixWritable.readMatrix(input));     
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeDouble(learningRate);
		output.writeDouble(l1Penalty);
		output.writeDouble(l2Penalty);
		output.writeInt(numberOfAtoms);
		output.writeInt(numberOfFeatures);
		this.transformer.write(output);;
		MatrixWritable.writeMatrix(output, dictionaryMatrix);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		DictionaryLearner other = (DictionaryLearner) obj;
		if (Double.doubleToLongBits(l1Penalty) != Double.doubleToLongBits(other.l1Penalty))
			return false;
		if (Double.doubleToLongBits(l2Penalty) != Double.doubleToLongBits(other.l2Penalty))
			return false;
		if (Double.doubleToLongBits(learningRate) != Double.doubleToLongBits(other.learningRate))
			return false;
		if (numberOfAtoms != other.numberOfAtoms)
			return false;
		if (numberOfFeatures != other.numberOfFeatures)
			return false;

		if (this.dictionaryMatrix.rowSize() != other.dictionaryMatrix.rowSize()) {
			return false;
		}
		if (this.dictionaryMatrix.columnSize() != other.dictionaryMatrix.columnSize()) {
			return false;
		}
		for (int i=0; i < this.dictionaryMatrix.rowSize(); i++) {
			if (!this.dictionaryMatrix.viewRow(i).equals(other.dictionaryMatrix.viewRow(i))) {
				return false;
			}
		}
		return true;
	}

}
