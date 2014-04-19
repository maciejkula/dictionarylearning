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
	private int regularizationStep = 10;
	private int regularizationCounter = 0;

	private int numberOfAtoms;
	private int numberOfFeatures;

	private final Transformer transformer;
	private Matrix dictionaryMatrix;

	public DictionaryLearner(int numberOfAtoms, int numberOfFeatures, Transformer transformer) {
		this.numberOfAtoms = numberOfAtoms;
		this.numberOfFeatures = numberOfFeatures;

		this.transformer = transformer;
		this.dictionaryMatrix = this.createEmptyDictionaryMatrix();

		System.out.println(String.format("Number of atoms %s", this.numberOfAtoms));
		System.out.println(String.format("Number of features %s", this.numberOfFeatures));
		System.out.println(String.format("Rows %s, columns %s", this.dictionaryMatrix.numRows(), this.dictionaryMatrix.numCols()));
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

	public int getRegularizationStep() {
		return regularizationStep;
	}

	public void setRegularizationStep(int regularizationStep) {
		this.regularizationStep = regularizationStep;
	}

	private Matrix createEmptyDictionaryMatrix() {
		// return new SparseRowMatrix(this.numberOfFeatures, this.numberOfAtoms);
		return new SparseColumnMatrix(this.numberOfFeatures, this.numberOfAtoms);
	}

	public Vector transform(Vector datapoint) {
		return this.transformer.transform(datapoint, this.dictionaryMatrix);
	}

	public Vector inverseTransform(Vector datapoint) {
		return this.transformer.inverseTransform(datapoint, this.dictionaryMatrix);
	}

	public void trainz(Vector datapoint) {
		this.initializeAtoms(datapoint);

		Vector projection = this.transformer.transform(datapoint, this.dictionaryMatrix);
		System.out.println("Computed projection");

		for (int i=0; i < this.numberOfFeatures; i++) {
			Vector featureRow = this.dictionaryMatrix.viewRow(i);
			// System.out.println("computing");
			double datapointValue = datapoint.get(i);
			// Enumerate over components
			for (int j=0; j < this.numberOfAtoms; j++) {
				double value = featureRow.get(j);
				double gradient = projection.get(j) * (value - datapointValue) + this.l2Penalty * value + this.l1Penalty * Math.signum(value);
				value = value - this.learningRate * gradient;
				if (Math.abs(value) < this.l1Penalty) {
					featureRow.set(j, 0.0);
				} else {
					featureRow.set(j, value);
				}
			}
		}
	}
	
	public void reproject() {
		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector atom = this.dictionaryMatrix.viewColumn(i);
			double atomL2Norm = atom.norm(2);
			atom.times(1/(Math.max(atomL2Norm, 1)));
		}
	}

	public void regularize() {
		
		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector atom = this.dictionaryMatrix.viewColumn(i);
			List<Integer> indicesToRemove = new ArrayList<Integer>();
			for (Element elem : atom.nonZeroes()) {
				// System.out.println(this.l2Penalty);
				double regularizedValue = (elem.get()
						* Math.pow((1 - this.learningRate * this.l2Penalty), this.regularizationCounter)
						- this.regularizationCounter * this.l1Penalty * Math.signum(elem.get()));
//				double regularizedValue = elem.get() -(this.regularizationCounter * this.learningRate 
//						* (elem.get() 
//								* this.l2Penalty
//								+ this.l1Penalty * Math.signum(elem.get())));
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
	
	private void incrementRegularizationCounter() {
		this.regularizationCounter++;
	}
	
	private void resetRegularizationCounter() {
		this.regularizationCounter = 0;
	}

	public void train(Vector datapoint) {
		this.initializeAtoms(datapoint);
		
		this.incrementRegularizationCounter();

		System.out.println("projecting started");
		Vector projection = this.transformer.transform(datapoint, this.dictionaryMatrix);
		System.out.println("projecting stopped");

		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector atom = this.dictionaryMatrix.viewColumn(i);
			double projectionWeight = projection.get(i);
			Vector difference = atom.minus(datapoint);
			for (Element elem : difference.nonZeroes()) {
				atom.incrementQuick(elem.index(), -this.learningRate * projectionWeight * elem.get());
			}
		}

		if (this.regularizationCounter % this.regularizationStep == 0) {
			System.out.println("regularizing");
			this.regularize();
			this.resetRegularizationCounter();
		}

		//    		for (int j=0; j < this.numberOfFeatures; j++) {
		//    			double value = atom.get(j);
		//    			//atom.assign(this.computeGradient(atom, datapoint).times(-1 * this.learningRate * projectionWeight));
		//    			double gradient = projectionWeight * (value - datapoint.get(j)) + this.l2Penalty * value + this.l1Penalty * Math.signum(value);
		//    			value = value - this.learningRate * gradient;
		//    			if (Math.abs(value) < this.l1Penalty) {
		//    				atom.set(j, 0.0);
		//    			} else {
		//    				atom.set(j, value);
		//    			}
		//    		}
	}


	private void initializeAtoms(Vector datapoint) {
		for (int i=0; i < this.numberOfAtoms; i++) {
			Vector column =  this.dictionaryMatrix.viewColumn(i);
			// System.out.println(column);
			if (column.getNumNonZeroElements() == 0) {
				System.out.println("Replacing an atom");
				column.assign(datapoint);
				break;
			}
		}
	}

	public Matrix getDictionary() {
		return this.dictionaryMatrix;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		this.setLearningRate(input.readDouble());
		this.setL1Penalty(input.readDouble());
		this.setL2Penalty(input.readDouble());
		this.numberOfAtoms = input.readInt();
		this.numberOfFeatures = input.readInt();
		this.transformer.readFields(input);
		this.dictionaryMatrix = MatrixWritable.readMatrix(input);     
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeDouble(learningRate);
		output.writeDouble(l1Penalty);
		output.writeDouble(l2Penalty);
		output.writeInt(numberOfAtoms);
		output.writeInt(numberOfFeatures);
		this.transformer.write(output);
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
