package com.github.maciejkula.dictionarylearning;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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

    private static class RowInnerProductsRunnable implements Callable<Void> {

        int rowIndex;
        SparseColumnMatrix inputMatrix;
        Matrix outputMatrix;

        public RowInnerProductsRunnable(int rowIndex, SparseColumnMatrix inputMatrix, Matrix outputMatrix) {
            this.rowIndex = rowIndex;
            this.inputMatrix = inputMatrix;
            this.outputMatrix = outputMatrix;
        }

        @Override
        public Void call() {
            Vector row =  this.inputMatrix.viewColumn(this.rowIndex);
            for (int j=this.rowIndex; j < this.inputMatrix.numCols(); j++) {
                Vector column = this.inputMatrix.viewColumn(j);
                double value = row.dot(column);
                this.outputMatrix.setQuick(this.rowIndex, j, value);
                this.outputMatrix.setQuick(j, this.rowIndex, value);   
            }
            return null;
        }
    }

    /*
     * Used for computing the left-hand side of the dictionary
     * projection linear system.
     * 
     * Returns the dense result of A'A. 
     */
    public static Matrix transposedDictionaryTimesDictionary(SparseColumnMatrix dictionary) {

        ExecutorService executor = Executors.newFixedThreadPool(4);

        Matrix output = new DenseMatrix(dictionary.numCols(), dictionary.numCols());
        List <RowInnerProductsRunnable> tasks = new ArrayList<RowInnerProductsRunnable>(dictionary.numCols());

        for (int i=0; i < dictionary.numCols(); i++) {
            tasks.add(new RowInnerProductsRunnable(i, dictionary, output));
        }

        try {
            executor.invokeAll(tasks);
        } catch (InterruptedException e) {
           throw new RuntimeException(e);
        }
        executor.shutdown();

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
