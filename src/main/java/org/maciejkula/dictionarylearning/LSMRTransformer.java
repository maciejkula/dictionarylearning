package org.maciejkula.dictionarylearning;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.CholeskyDecomposition;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.solver.ConjugateGradientSolver;
import org.apache.mahout.math.solver.LSMR;

public class LSMRTransformer implements Transformer, Writable {

    private final LSMR solver;

    public LSMRTransformer() {
        this.solver = new LSMR();
    }
    


    @Override
    public Vector transform(Vector datapoint, Matrix dictionary) {
    	// return dictionary.viewRow(0);
//    	Vector projection = new DenseVector(dictionary.numCols());
//    	for (int i=0; i < dictionary.numCols(); i++) {
//    		Vector candidates = new DenseVector(dictionary.numCols());
//    		for (int j=0; j < dictionary.numCols(); j++) {
//    			candidates.setQuick(j, datapoint.dot(dictionary.viewColumn(j)));
//    		}
//    		int closestIndex = candidates.maxValueIndex();
//    		projection.setQuick(closestIndex, candidates.getQuick(closestIndex));
//    		datapoint = datapoint.minus(dictionary.viewColumn(i).times(projection.get(closestIndex)));
//    	}
//    	System.out.println(projection);
//        return projection; 
    	// System.out.println("Multiplying");
    	Long start = System.currentTimeMillis();
    	//dictionary.times(dictionary.transpose());
    	// System.out.println(dictionary.transpose().times(dictionary));
    	// System.out.println(MathUtils.squareMatrix(dictionary));
    	// System.out.println(String.format("Finished multiplying in %s", System.currentTimeMillis() - start));
    	
    	// CholeskyDecomposition decomp = new CholeskyDecomposition(MathUtils.squareMatrix(dictionary));
    	// return decomp.solveLeft(dictionary.transpose().times(datapoint));
    	
    	// ConjugateGradientSolver cjsolver = new ConjugateGradientSolver();
    	// Vector projection = cjsolver.solve(MathUtils.squareMatrix(dictionary), dictionary.transpose().times(datapoint));
    	// return solver.solve(MathUtils.squareMatrix(dictionary), dictionary.transpose().times(datapoint));
    	// return this.solver.solve(dictionary, datapoint);
    	
    	Matrix datapointMatrix = new SparseColumnMatrix(datapoint.size(),1);
    	datapointMatrix.assignColumn(0, datapoint);
    	
    	// CholeskyDecomposition decomp = new CholeskyDecomposition(MathUtils.squareMatrix(dictionary));
    	
    	Matrix Amatrix = MathUtils.squareMatrix(dictionary); // dictionary.transpose().times(dictionary); //
    	// for (int i=0; i < Amatrix.numRows(); i++) {
    	// 	Amatrix.setQuick(i, i, Amatrix.getQuick(i, i) + 0.00000);
    	// }
    	// System.out.println(Amatrix);
    	// dictionary.transpose().times(datapointMatrix)
    	// Vector projection = new QRDecomposition(Amatrix).solve(MathUtils.transposedMatrixVectorMultiplication(dictionary, datapoint)).viewColumn(0);
    	//System.out.println(projection);
    	//System.out.println(this.solver.solve(dictionary, datapoint));
    	//return projection;
    	return this.solver.solve(Amatrix, MathUtils.transposedMatrixVectorMultiplication(dictionary, datapoint).viewColumn(0));
        // return this.solver.solve(dictionary, datapoint);
    }

    @Override
    public Vector inverseTransform(Vector projection, Matrix dictionary) {
    	return MathUtils.reproject(dictionary, projection);
        // return MathUtils.transposedMatrixVectorMultiplication(dictionary, projection).viewColumn(0);//dictionary.times(projection);
    }

    @Override
    public void readFields(DataInput arg0) throws IOException {
        // No state.
    }

    @Override
    public void write(DataOutput output) throws IOException {
        // No state.
    }

}
