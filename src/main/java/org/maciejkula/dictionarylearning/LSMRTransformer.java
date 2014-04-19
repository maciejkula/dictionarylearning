package org.maciejkula.dictionarylearning;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.solver.LSMR;

public class LSMRTransformer implements Transformer, Writable {

    private final LSMR solver;

    public LSMRTransformer() {
        this.solver = new LSMR();
    }

    @Override
    public Vector transform(Vector datapoint, Matrix dictionary) {
    	return this.solver.solve(MathUtils.transposedDictionaryTimesDIctionary(dictionary), 
    			MathUtils.transposedDictionaryTimesDatapoint(dictionary, datapoint).viewColumn(0));
    }

    @Override
    public Vector inverseTransform(Vector projection, Matrix dictionary) {
    	return MathUtils.inverseTransform(dictionary, projection);
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
