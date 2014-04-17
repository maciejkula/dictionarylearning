package org.maciejkula.dictionarylearning;

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
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.csv.CSVVectorIterator;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;

public class DictionaryLearningTestCase extends TestCase {
    
    public Matrix readData() {
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

        SparseRowMatrix matrix = new SparseRowMatrix(vectors.size(), vectors.get(0).size(), vectors.toArray(new Vector[3]));
        
        return matrix;
    }

    public void testAccuracy() {

        Matrix matrix = readData();
        DictionaryLearner dictionaryLearner = new DictionaryLearner(10, matrix.columnSize(), new LSMRTransformer());
        dictionaryLearner.setL1Norm(0.15);


        System.out.println("doing stuff");

        for (Vector row : matrix) {
            dictionaryLearner.train(row);
            System.out.println(dictionaryLearner.transform(row));
            System.out.println(dictionaryLearner.inverseTransform(dictionaryLearner.transform(row)));
            System.out.println(row);
        }

        double squareError = 0.0;
        for (Vector datapoint : matrix) {
            squareError = squareError + datapoint.getDistanceSquared(dictionaryLearner.inverseTransform(dictionaryLearner.transform(datapoint)));
        }
        System.out.println(String.format("Error: %s", squareError));

        System.out.println("done");
    }
    
    public void testSerialization() {
        
        Matrix data = readData();        
        DictionaryLearner dictionaryLearner = new DictionaryLearner(10, data.columnSize(), new LSMRTransformer());
        dictionaryLearner.setL1Norm(0.15);
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
        
        this.assertTrue(dictionaryLearner.equals(deserializedLearner));
        
    }



}
