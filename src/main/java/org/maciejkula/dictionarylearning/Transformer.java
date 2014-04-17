package org.maciejkula.dictionarylearning;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

public interface Transformer extends Writable{
    
    public Vector transform(Vector datapoint, Matrix dictionary);
    public Vector inverseTransform(Vector projection, Matrix dictionary);
    

}
