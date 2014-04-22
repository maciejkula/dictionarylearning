package com.github.maciejkula.dictionarylearning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

public class EvaluationUtils {

    private static class ElementComparator implements Comparator<Element> {
        @Override
        public int compare(Element a, Element b) {
            return a.get() < b.get() ? 1 : a.get() == b.get() ? 0 : -1;
        }
    }

    public static class RankMap {

        private final Map<Integer, Integer> ranks;
        private Integer rankOfZeroElement;

        public RankMap(int capacity) {
            this.ranks = new HashMap<Integer, Integer>(capacity);
        }

        public Integer get(int index) {
            Integer value = this.ranks.get(index);      
            if (value != null) {
                return value;
            } else {
                return this.rankOfZeroElement;
            }
        }

        public void put(int index, int value) {
            this.ranks.put(index, value);
        }

        public void setRankOfZeroElement(int value) {
            this.rankOfZeroElement = value;
        }

    }

    public static RankMap getRankMap(Vector scores) {
        List<Element> elements = new ArrayList<Element>(scores.getNumNonZeroElements());
        for (Element elem : scores.nonZeroes()) {
            elements.add(scores.getElement(elem.index()));
        }
        Collections.sort(elements, new ElementComparator());

        RankMap rankMap = new RankMap(scores.getNumNonZeroElements() + 1);
        int numZeroes = scores.size() - scores.getNumNonZeroElements();
        int numAboveZero = 0;

        for (int i=0; i < elements.size(); i++) {
            Element elem = elements.get(i);
            if (elem.get() > 0.0) {
                rankMap.put(elem.index(), i);
                numAboveZero++;
            }
            else {
                rankMap.put(elem.index(), numZeroes + i);
            }
        }
        rankMap.setRankOfZeroElement(numAboveZero);

        return rankMap;
    }

}
