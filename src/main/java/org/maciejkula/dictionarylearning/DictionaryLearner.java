package org.maciejkula.dictionarylearning;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;

public class DictionaryLearner implements Writable {

    private double learningRate = 0.01;
    private double l1 = 0.0;
    private double l2 = 0.0;

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
    
    public void setL1Norm(double norm) {
        this.l1 = norm;
    }
    
    public double getL1Norm() {
        return this.l1;
    }
    
    public void setL2Norm(double norm) {
        this.l2 = norm;
    }
    
    public double getL2Norm() {
        return this.l2;
    }

    private Matrix createEmptyDictionaryMatrix() {
        return new SparseRowMatrix(this.numberOfFeatures, this.numberOfAtoms);
    }
    
    public Vector transform(Vector datapoint) {
        return this.transformer.transform(datapoint, this.dictionaryMatrix);
    }
    
    public Vector inverseTransform(Vector datapoint) {
        return this.transformer.inverseTransform(datapoint, this.dictionaryMatrix);
    }

    public void train(Vector datapoint) {
        this.initializeAtoms(datapoint);

        Vector projection = this.transformer.transform(datapoint, this.dictionaryMatrix);

        for (int i=0; i < this.numberOfFeatures; i++) {
            Vector featureRow = this.dictionaryMatrix.viewRow(i);
            double datapointValue = datapoint.get(i);
            // Enumerate over components
            for (int j=0; j < this.numberOfAtoms; j++) {
                double value = featureRow.get(j);
                double gradient = projection.get(j) * (value - datapointValue) + this.l2 * value + this.l1 * Math.signum(value);
                value = value - this.learningRate * gradient;
                if (Math.abs(value) < this.l1) {
                    featureRow.set(j, 0.0);
                } else {
                    featureRow.set(j, value);
                }
            }
        }
    }

    private void initializeAtoms(Vector datapoint) {
        for (int i=0; i < this.numberOfAtoms; i++) {
            Vector column =  this.dictionaryMatrix.viewColumn(i);
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
        this.setL1Norm(input.readDouble());
        this.setL2Norm(input.readDouble());
        this.numberOfAtoms = input.readInt();
        this.numberOfFeatures = input.readInt();
        this.transformer.readFields(input);
        this.dictionaryMatrix = MatrixWritable.readMatrix(input);     
    }

    @Override
    public void write(DataOutput output) throws IOException {
        output.writeDouble(learningRate);
        output.writeDouble(l1);
        output.writeDouble(l2);
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
        if (Double.doubleToLongBits(l1) != Double.doubleToLongBits(other.l1))
            return false;
        if (Double.doubleToLongBits(l2) != Double.doubleToLongBits(other.l2))
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
