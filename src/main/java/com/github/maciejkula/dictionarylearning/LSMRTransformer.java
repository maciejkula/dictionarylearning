package com.github.maciejkula.dictionarylearning;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.solver.LSMR;

public class LSMRTransformer implements Transformer, Writable {

    private final LSMR solver;
    
    private Matrix transposedDictionaryTimesDictionaryCache;

    public LSMRTransformer() {
        this.solver = new LSMR();
    }

    /*
     * Solves the linear projection A'Ax = A'y, where A is the dictionary
     * and y is the datapoint.
     */
    @Override
    public Vector transform(Vector datapoint, SparseColumnMatrix dictionary) {
    	return this.solver.solve(this.getTransposedDictionaryTimesDictionary(dictionary), 
    			MathUtils.transposedDictionaryTimesDatapoint(dictionary, datapoint));
    }

    /*
     * Reconstructs the datapoint y from its projection x on the dictionary A.
     */
    @Override
    public Vector inverseTransform(Vector projection, SparseColumnMatrix dictionary) {
    	return MathUtils.inverseTransform(dictionary, projection);
    }
    
    @Override
    public void clearCaches() {
    	this.transposedDictionaryTimesDictionaryCache = null;
    }
    
    private Matrix getTransposedDictionaryTimesDictionary(SparseColumnMatrix dictionary) {
    	if (this.transposedDictionaryTimesDictionaryCache == null) {
    		this.transposedDictionaryTimesDictionaryCache = MathUtils.transposedDictionaryTimesDictionary(dictionary);
    	}
    	return this.transposedDictionaryTimesDictionaryCache;
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
