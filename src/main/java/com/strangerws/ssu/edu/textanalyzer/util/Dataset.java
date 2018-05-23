package com.strangerws.ssu.edu.textanalyzer.util;

import java.util.ArrayList;
import java.util.List;

public class Dataset<D extends Object, R extends Object> {
    private List<Entry<D, R>> data;

    public void setDataset(List<D> data, List<R> result){
        if(data.size() != result.size()){
            throw new RuntimeException("arrays length do not match each other");
        }

        for (int i = 0; i < data.size(); i++) {
            appendEntry(data.get(i), result.get(i));
        }

    }

    public Dataset() {
        this.data = new ArrayList<>();
    }

    public void appendEntry(D data, R result){
        this.data.add(new Entry<>(data, result));
    }
    public void appendEntry(Entry entry){
        this.data.add(entry);
    }

    public int size(){
        return data.size();
    }

    public Entry<D, R> get(int index){
        return data.get(index);
    }

    public static class Entry<D, R>{
        private D data;
        private R result;

        public Entry(D data, R result) {
            this.data = data;
            this.result = result;
        }

        public D getData() {
            return data;
        }

        public void setData(D data) {
            this.data = data;
        }

        public R getResult() {
            return result;
        }

        public void setResult(R result) {
            this.result = result;
        }
    }
}
