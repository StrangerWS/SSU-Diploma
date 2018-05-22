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
    private List<Layer> mLayers;

    public List<Layer> getLayers() {
        return mLayers;
    }

    public LayerBuilder() {
        mLayers = new ArrayList<Layer>();
    }

    public LayerBuilder(Layer layer) {
        this();
        mLayers.add(layer);
    }

    public LayerBuilder addLayer(Layer layer) {
        mLayers.add(layer);
        return this;
    }
}