package com.strangerws.ssu.edu.textanalyzer;


import CNN.src.edu.hitsz.c102c.dataset.Dataset;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.LayerBuilder;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.NeuralNet;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.element.Layer;

public class Main {
    public static void main(String[] args) {
        LayerBuilder builder = new LayerBuilder();
        builder
                .addLayer(Layer.buildInputLayer(new Size(28, 28)))
                .addLayer(Layer.buildConvolutionalLayer(6, new Size(5, 5)))
                .addLayer(Layer.buildSampleLayer(new Size(2,2)))
                .addLayer(Layer.buildConvolutionalLayer(12, new Size(5,5)))
                .addLayer(Layer.buildSampleLayer(new Size(2, 2)))
                .addLayer(Layer.buildOutputLayer(10));
        NeuralNet cnn = new NeuralNet(builder, 50);

        String fileName = "src/main/resources/dataset/train.format";
        Dataset dataset = Dataset.load(fileName, ",", 784);
        cnn.train(dataset, 3);
        String modelName = "src/main/resources/model/model.cnn";
        cnn.saveModel(modelName);
        dataset.clear();
        dataset = null;

        Dataset testset = Dataset.load("src/main/resources/dataset/test.format", ",", -1);
        cnn.predict(testset, "src/main/resources/dataset/test.predict");
    }

}
