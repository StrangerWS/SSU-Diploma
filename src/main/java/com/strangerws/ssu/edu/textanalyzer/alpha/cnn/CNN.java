package com.strangerws.ssu.edu.textanalyzer.alpha.cnn;

import com.strangerws.ssu.edu.textanalyzer.neuralnet.api.LayerType;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.element.Layer;
import com.strangerws.ssu.edu.textanalyzer.util.Dataset;

import java.util.List;

public class CNN {
    private List<Layer> layers;
    private Dataset dataset;

    public CNN(List<Layer> layers) {
        this.layers = layers;
    }

    private double[] normalizeInput(byte[] array){
        double[] normalized = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            normalized[i] = (array[i] + 128)/255;
        }
        return normalized;
    }

    private void calculateOutputError(){
        Layer output = layers.get(layers.size() - 1);
        if (output.getType() != LayerType.OUTPUT) throw new RuntimeException("last layer is not output!");
        if (output.getOutputCount() != dataset.size()) throw new RuntimeException("output count is not equals to dataset size!");
        int[] expected = new int[dataset.size()];
        for (int i = 0; i < dataset.size(); i++) {
            expected[i] = dataset.get(i).getResult();
        }
    }
}
