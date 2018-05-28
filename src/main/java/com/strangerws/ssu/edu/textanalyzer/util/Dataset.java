package com.strangerws.ssu.edu.textanalyzer.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class Dataset {
    private List<Entry> data;

    public void setDataset(List<double[]> data, List<Integer> result) {
        if (data.size() != result.size()) {
            throw new RuntimeException("arrays length do not match each other");
        }

        for (int i = 0; i < data.size(); i++) {
            appendEntry(data.get(i), result.get(i));
        }
    }

    public void setDataset(List<double[]> data) {
        for (double[] dataEntry : data) {
            appendEntry(dataEntry, 0);
        }
    }

    public Iterator<Entry> iterator() {
        return data.iterator();
    }

    public Dataset() {
        this.data = new ArrayList<>();
    }

    public void shuffle() {
        Collections.shuffle(data);
    }

    public void appendEntry(double[] data, int result) {
        this.data.add(new Entry(data, result));
    }

    public void appendEntry(Entry entry) {
        this.data.add(entry);
    }

    public int size() {
        return data.size();
    }

    public Entry get(int index) {
        return data.get(index);
    }

    public static class Entry {
        private double[] data;
        private int result;

        public Entry(double[] data, int result) {
            this.data = data;
            this.result = result;
        }

        public double[] getData() {
            return data;
        }

        public void setData(double[] data) {
            this.data = data;
        }

        public int getResult() {
            return result;
        }

        public void setResult(Character result) {
            this.result = result;
        }
    }
}
