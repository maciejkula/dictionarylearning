package com.github.maciejkula.dictionarylearning;

import junit.framework.TestCase;

import org.apache.mahout.math.RandomAccessSparseVector;

public class EvaluationUtilsTestCase extends TestCase {
    
    public void testRankMapCreation() {
        RandomAccessSparseVector vector = new RandomAccessSparseVector(10);
        vector.setQuick(1, 10);
        vector.setQuick(2, 5);
        vector.setQuick(3, -2);
        
        EvaluationUtils.RankMap rankMap = EvaluationUtils.getRankMap(vector);
        
        assertTrue(rankMap.get(1) == 0);
        assertTrue(rankMap.get(2) == 1);
        assertTrue(rankMap.get(4) == 5);
        assertTrue(rankMap.get(3) == 9);
    }

}
