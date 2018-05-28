package com.strangerws.ssu.edu.textanalyzer.neuralnet;

import com.strangerws.ssu.edu.textanalyzer.neuralnet.element.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * The constructor
 * mode constructs
 * each layer, requiring
 * that the
 * penultimate layer
 * must be
 * the sampling
 * layer and
 * not the
 * convolution layer
 *
 * @author jiqunpeng
 * <p>
 * Created:2014-7-8 4:54:29PM
 */

public class LayerBuilder {
    private List<Layer> layers;

    public List<Layer> build() {
        return layers;
    }

    public LayerBuilder() {
        layers = new ArrayList<>();
    }

    public LayerBuilder(Layer layer) {
        this();
        layers.add(layer);
    }

    public LayerBuilder addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }
}