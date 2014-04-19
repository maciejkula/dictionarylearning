package org.maciejkula.dictionarylearning;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.Vector;

public interface Transformer extends Writable{
    
    public Vector transform(Vector datapoint, SparseColumnMatrix dictionary);
    public Vector inverseTransform(Vector projection, SparseColumnMatrix dictionary);
    

}
