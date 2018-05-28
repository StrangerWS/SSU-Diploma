package com.strangerws.ssu.edu.textanalyzer.neuralnet;


import CNN.src.edu.hitsz.c102c.util.ConcurenceRunner;
import CNN.src.edu.hitsz.c102c.util.Log;
import CNN.src.edu.hitsz.c102c.util.Util;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.element.Layer;
import com.strangerws.ssu.edu.textanalyzer.util.Dataset;
import com.strangerws.ssu.edu.textanalyzer.util.Utils;
import com.strangerws.ssu.edu.textanalyzer.util.Utils.Operator;

import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class NeuralNet implements Serializable {
    /**
     *
     */
    private static double ALPHA = 0.85;
    protected static final double LAMBDA = 0;
    // The layers of the network
    private List<Layer> layers;
    // layer number
    private int layerNum;

    // The size of the batch update
    private int batchSize;
    // Divisor operator, divides each element of the matrix by a value
    private Operator divideBatchSize;

    // Multiplier operator, multiplying each element of the matrix by an alpha value
    private Operator multiplyAlpha;

    // Multiplier operator, multiplying each element of the matrix by 1-labmda*alpha
    private Operator multiplyLambda;

    private static AtomicBoolean stopTrain;


    /**
     * Initialize the network
     *
     * @param layers Network layer
     */
    public NeuralNet(List<Layer> layers, final int batchSize) {
        this.layers = layers;
        layerNum = layers.size();
        this.batchSize = batchSize;
        setup(batchSize);
        initOperator();
    }

    /**
     * Initialization operator
     */
    private void initOperator() {
        divideBatchSize = (double value) -> value / batchSize;
        multiplyAlpha = (double value) -> value * ALPHA;
        multiplyLambda = (double value) -> value * (1 - LAMBDA * ALPHA);
    }

    /**
     * Train the network on the training set
     *
     * @param trainset
     * @param repeat   Number of iterations
     */
    public void train(Dataset trainset, int repeat) {
        // Monitor stop button
        new Listener().start();
        for (int t = 0; t < repeat && !stopTrain.get(); t++) {
            int epochsNum = trainset.size() / batchSize;
            if (trainset.size() % batchSize != 0)
                epochsNum++; // More than one extraction, ie rounding up
            Log.i("");
            Log.i(t + "th iterator epochsNum:" + epochsNum);
            int right = 0;
            int count = 0;
            for (int i = 0; i < epochsNum; i++) {
                int[] randPerm = Utils.randomPerm(trainset.size(), batchSize);
                Layer.resetBatch();

                for (int index : randPerm) {
                    boolean isRight = train(trainset.get(index));
                    if (isRight)
                        right++;
                    count++;
                    Layer.incrementBatch();
                }

                // Update weight after running a batch
                updateParas();
                if (i % 50 == 0) {
                    System.out.print("..");
                    if (i + 50 > epochsNum)
                        System.out.println();
                }
            }
            double p = 1.0 * right / count;
            if (t % 10 == 1 && p > 0.96) { // Adjust the quasi-learning rate dynamically
                ALPHA = 0.001 + ALPHA * 0.9;
                Log.i("Set alpha = " + ALPHA);
            }
            Log.i("precision " + right + "/" + count + "=" + p);
        }
    }

    static class Listener extends Thread {
        Listener() {
            setDaemon(true);
            stopTrain = new AtomicBoolean(false);
        }

        @Override
        public void run() {
            System.out.println("Input & to stop train.");
            while (true) {
                try {
                    int a = System.in.read();
                    if (a == '&') {
                        stopTrain.compareAndSet(false, true);
                        break;
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("Listener stop");
        }

    }

    /**
     * Test Data
     *
     * @param trainset
     * @return
     */

    public double test(Dataset trainset) {
        Layer.resetBatch();
        Iterator<Dataset.Entry> iter = trainset.iterator();
        int right = 0;
        while (iter.hasNext()) {
            Dataset.Entry record = iter.next();
            forward(record);
            Layer outputLayer = layers.get(layerNum - 1);
            int mapNum = outputLayer.getOutputCount();
            double[] out = new double[mapNum];
            for (int m = 0; m < mapNum; m++) {
                double[][] outmap = outputLayer.getMap(m);
                out[m] = outmap[0][0];
            }
            if ((int) record.getResult() == Utils.getMaxIndex(out))
                right++;
        }
        double p = 1.0 * right / trainset.size();
        Log.i("precision", p + "");
        return p;
    }

    /**
     * forecast result
     *
     * @param testset
     * @param fileName
     */

    public void predict(Dataset testset, String fileName) {
        Log.i("begin predict");
        try {
            int max = layers.get(layerNum - 1).getClassNum();
            PrintWriter writer = new PrintWriter(new File(fileName));
            Layer.resetBatch();
            Iterator<Dataset.Entry> iter = testset.iterator();
            while (iter.hasNext()) {
                Dataset.Entry record = iter.next();
                forward(record);
                Layer outputLayer = layers.get(layerNum - 1);

                int mapNum = outputLayer.getOutputCount();
                double[] out = new double[mapNum];
                for (int m = 0; m < mapNum; m++) {
                    double[][] outmap = outputLayer.getMap(m);
                    out[m] = outmap[0][0];
                }
                int lable = Utils.getMaxIndex(out);
                writer.write(lable + "\n");
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Log.i("end predict");
    }

    private boolean isSame(double[] output, double[] target) {
        boolean r = true;
        for (int i = 0; i < output.length; i++)
            if (Math.abs(output[i] - target[i]) > 0.5) {
                r = false;
                break;
            }

        return r;
    }

    /**
     * Train a
     * entry while
     * returning whether
     * the correct
     * current entry
     * is predicted
     *
     * @param entry
     * @return
     */

    private boolean train(Dataset.Entry entry) {
        forward(entry);
        return backPropagation(entry);
        // System.exit(0);
    }

        /*
                *
    Reverse transmission
        */

    private boolean backPropagation(Dataset.Entry record) {
        boolean result = setOutLayerErrors(record);
        setHiddenLayerErrors();
        return result;
    }

    /**
     * Update parameters
     */

    private void updateParas() {
        for (int l = 1; l < layerNum; l++) {
            Layer layer = layers.get(l);
            Layer lastLayer = layers.get(l - 1);
            switch (layer.getType()) {
                case CONVOLUTIONAL:
                case OUTPUT:
                    updateKernels(layer, lastLayer);
                    updateBias(layer);
                    break;
                default:
                    break;
            }
        }
    }

    /**
     * Update bias
     *
     * @param layer
     */

    private void updateBias(final Layer layer) {
        final double[][][][] errors = layer.getErrors();
        int mapNum = layer.getOutputCount();

        new ConcurenceRunner.TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    double[][] error = Utils.sum(errors, j);
                    // Update bias
                    double deltaBias = Utils.sum(error) / batchSize;
                    double bias = layer.getBias(j) + ALPHA * deltaBias;
                    layer.setBias(j, bias);
                }
            }
        }.start();

    }

    /**
     * Update layer
     * layer convolution
     * <p>
     * kernel(weight) and offset
     *
     * @param layer     Current layer
     * @param lastLayer Previous layer
     */

    private void updateKernels(final Layer layer, final Layer lastLayer) {
        int mapNum = layer.getOutputCount();
        final int lastMapNum = lastLayer.getOutputCount();
        new ConcurenceRunner.TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    for (int i = 0; i < lastMapNum; i++) {
                        // Sum the delta for each record of the batch
                        double[][] deltaKernel = null;
                        for (int r = 0; r < batchSize; r++) {
                            double[][] error = layer.getError(r, j);
                            if (deltaKernel == null)
                                deltaKernel = Utils.convnValid(
                                        lastLayer.getMap(r, i), error);
                            else { // cumulative summation
                                deltaKernel = Utils.matrixOp(Utils.convnValid(
                                        lastLayer.getMap(r, i), error),
                                        deltaKernel, null, null, Utils.plus);
                            }
                        }

                        // Divide by batchSize
                        deltaKernel = Utils.matrixOp(deltaKernel,
                                divideBatchSize);
                        // Update the convolution kernel
                        double[][] kernel = layer.getKernel(i, j);
                        deltaKernel = Utils.matrixOp(kernel, deltaKernel,
                                multiplyLambda, multiplyAlpha, Utils.plus);
                        layer.setKernel(i, j, deltaKernel);
                    }
                }

            }
        }.start();

    }

    /**
     * Set the
     * residual of
     * each layer
     * in the
     * setting
     */

    private void setHiddenLayerErrors() {
        for (int l = layerNum - 2; l > 0; l--) {
            Layer layer = layers.get(l);
            Layer nextLayer = layers.get(l + 1);
            switch (layer.getType()) {
                case SAMPLE:
                    setSampleErrors(layer, nextLayer);
                    break;
                case CONVOLUTIONAL:
                    setConvolutionalErrors(layer, nextLayer);
                    break;
                default:
                    // Only the sampling and convolution layers need to process the residuals, the input layer has no residuals, and the output layer has already processed
                    break;
            }
        }
    }

    /**
     * Set the
     * sampling layer
     * residual
     *
     * @param layer
     * @param nextLayer
     */

    private void setSampleErrors(final Layer layer, final Layer nextLayer) {
        int mapNum = layer.getOutputCount();
        final int nextMapNum = nextLayer.getOutputCount();
        new ConcurenceRunner.TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    double[][] sum = null; // Summing each convolution
                    for (int j = 0; j < nextMapNum; j++) {
                        double[][] nextError = nextLayer.getError(j);
                        double[][] kernel = nextLayer.getKernel(i, j);
                        // A 180 degree rotation of the convolution kernel and convolution in full mode
                        if (sum == null)
                            sum = Utils.convnFull(nextError, Utils.rot180(kernel));
                        else
                            sum = Utils.matrixOp(
                                    Utils.convnFull(nextError,
                                            Utils.rot180(kernel)), sum, null,
                                    null, Utils.plus);
                    }
                    layer.setError(i, sum);
                }
            }

        }.start();

    }

    /**
     * Set the
     * residual of
     * the convolutional
     * layer
     *
     * @param layer
     * @param nextLayer
     */

    private void setConvolutionalErrors(final Layer layer, final Layer nextLayer) {
        // The next layer of the convolutional layer is the sampling layer, that is, the number of maps in the two layers is the same, and a map is only connected to a map that makes one layer.
        // so we only need to expand the residual kronecker of the next level and use the dot product
        int mapNum = layer.getOutputCount();
        new ConcurenceRunner.TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int m = start; m < end; m++) {
                    Size scale = nextLayer.getScaleSize();
                    double[][] nextError = nextLayer.getError(m);
                    double[][] map = layer.getMap(m);
                    // Matrix multiplication, but 1-value operation for each element value of the second matrix
                    double[][] outMatrix = Utils.matrixOp(map,
                            Utils.cloneMatrix(map), null, Utils.one_value,
                            Utils.multiply);
                    outMatrix = Utils.matrixOp(outMatrix,
                            Utils.kronecker(nextError, scale), null, null,
                            Utils.multiply);
                    layer.setError(m, outMatrix);
                }

            }

        }.start();

    }

    /**
     * Set the
     * residual value
     * of the
     * output layer, the
     * number of
     * nerve cells
     * in the
     * output layer
     * is less, temporarily do
     * not consider
     * multi-threaded
     *
     * @param record
     * @return
     */

    private boolean setOutLayerErrors(Dataset.Entry record) {
        Layer outputLayer = layers.get(layerNum - 1);
        int mapNum = outputLayer.getOutputCount();

        double[] target = new double[mapNum];
        double[] outmaps = new double[mapNum];
        for (int m = 0; m < mapNum; m++) {
            double[][] outmap = outputLayer.getMap(m);
            double sum = 0;
            for (double[] anOutmap : outmap) {
                for (double anAnOutmap : anOutmap) {
                    sum += anAnOutmap;
                }
            }
            outmaps[m] = sum;

        }
        int resultId = (int) record.getResult();//getting number of suggested neuron
        target[resultId] = 1;//setting max response value to target
        for (int m = 0; m < mapNum; m++) {
            outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m])
                    * (target[m] - outmaps[m]));
        }
        System.out.println("expected: " + resultId + "\tactual: " + Utils.getMaxIndex(outmaps));
        return resultId == Utils.getMaxIndex(outmaps);
    }

    /**
     * Forward calculation
     * of a
     * entry
     *
     * @param entry
     */

    private void forward(Dataset.Entry entry) {
        // Set the input layer's map
        setInLayerOutput(entry);
        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Layer lastLayer = layers.get(l - 1);
            switch (layer.getType()) {
                case CONVOLUTIONAL: // Calculates the output of the convolutional layer
                    setConvolutionalOutput(layer, lastLayer);
                    break;
                case SAMPLE: // Calculate the output of the sampling layer
                    setSampleOutput(layer, lastLayer);
                    break;
                case OUTPUT: // Calculate the output of the output layer, the output layer is a special convolutional layer
                    setConvolutionalOutput(layer, lastLayer);
                    break;
                default:
                    break;
            }
        }
    }

    /**
     * According to
     * the record
     * value,
     * set the
     * output value
     * of the
     * input layer
     *
     * @param entry
     */

    private void setInLayerOutput(Dataset.Entry entry) {
        final Layer inputLayer = layers.get(0);
        final Size mapSize = inputLayer.getSize();
        final double[] attr = entry.getData();
        if (attr.length != mapSize.x * mapSize.y)
            throw new RuntimeException(" The size of the data record does not match the defined map size! ");
        for (int i = 0; i < mapSize.x; i++) {
            for (int j = 0; j < mapSize.y; j++) {
                // Make the one-dimensional vector of the record attribute a two-dimensional matrix
                inputLayer.setMapValue(0, i, j, attr[mapSize.x * i + j]);
            }
        }
    }

    /**
     * Calculate the
     * convolutional output
     * value,
     * each thread
     * is responsible for
     * a part
     * of the
     * map
     */

    private void setConvolutionalOutput(final Layer layer, final Layer lastLayer) {
        int mapNum = layer.getOutputCount();
        final int lastMapNum = lastLayer.getOutputCount();
        new ConcurenceRunner.TaskManager(mapNum) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    double[][] sum = null; // Summing the convolution of each input map
                    for (int i = 0; i < lastMapNum; i++) {
                        double[][] lastMap = lastLayer.getMap(i);
                        double[][] kernel = layer.getKernel(i, j);
                        if (sum == null)
                            sum = Utils.convnValid(lastMap, kernel);
                        else
                            sum = Utils.matrixOp(
                                    Utils.convnValid(lastMap, kernel), sum,
                                    null, null, Utils.plus);
                    }
                    final double bias = layer.getBias(j);
                    sum = Utils.matrixOp(sum, new Operator() {
                        @Override
                        public double process(double value) {
                            return Utils.sigmod(value + bias);
                        }

                    });

                    layer.setMapValue(j, sum);
                }
            }

        }.start();

    }

    /**
     * Set the
     * sampling layer
     * output value, the
     * sampling layer
     * is the
     * mean of
     * the convolution
     * layer
     *
     * @param layer
     * @param lastLayer
     */

    private void setSampleOutput(final Layer layer, final Layer lastLayer) {
        int lastMapNum = lastLayer.getOutputCount();
        new ConcurenceRunner.TaskManager(lastMapNum) {

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    double[][] lastMap = lastLayer.getMap(i);
                    Size scaleSize = layer.getScaleSize();
                    // Mean processing by scaleSize area
                    double[][] sampMatrix = Utils.scaleMatrix(lastMap, scaleSize);
                    layer.setMapValue(i, sampMatrix);
                }
            }

        }.start();

    }

    /**
     * Set the
     * parameters of
     * each layer
     * of the
     * CNN network
     *
     * @param batchSize
     */

    public void setup(int batchSize) {
        Layer inputLayer = layers.get(0);
        // Each layer needs to initialize the output map
        inputLayer.initOutputMaps(batchSize);
        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            Layer frontLayer = layers.get(i - 1);
            int frontMapNum = frontLayer.getOutputCount();
            switch (layer.getType()) {
                case INPUT:
                    break;
                case CONVOLUTIONAL:
                    // Set the size of the map
                    layer.setSize(frontLayer.getSize().subtract(
                            layer.getKernelSize(), 1));
                    // Initialize the convolution kernel with a total of frontMapNum*outMapNum convolution kernels

                    layer.initKernel(frontMapNum);
                    // Initialize offset, total of frontMapNum*outMapNum offsets
                    layer.initBias();
                    // Each record of batch must maintain a residual
                    layer.initErros(batchSize);
                    // Each layer needs to initialize the output map
                    layer.initOutputMaps(batchSize);
                    break;
                case SAMPLE:
                    // The sampling layer has the same number of maps as the previous layer
                    layer.setOutputCount(frontMapNum);
                    // The size of the map layer is the size of the map above the size of the scale
                    layer.setSize(frontLayer.getSize().divide(
                            layer.getScaleSize()));
                    // Each record of batch must maintain a residual
                    layer.initErros(batchSize);
                    // Each layer needs to initialize the output map
                    layer.initOutputMaps(batchSize);
                    break;
                case OUTPUT:
                    //The initialization weight (convolution kernel), the output layer convolution kernel size is the map size of the previous layer
                    layer.initOutputKernel(frontMapNum, frontLayer.getSize());
                    // Initialize offset, total of frontMapNum*outMapNum offsets
                    layer.initBias();
                    // Each record of batch must maintain a residual
                    layer.initErros(batchSize);
                    // Each layer needs to initialize the output map
                    layer.initOutputMaps(batchSize);
                    break;
            }
        }
    }

    /**
     * Serialized save
     * model
     *
     * @param fileName
     */

    public void saveModel(String fileName) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(fileName));
            oos.writeObject(this);
            oos.flush();
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * Deserialize import model
     *
     * @param fileName
     * @return
     */

    public static NeuralNet loadModel(String fileName) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(
                    fileName));
            NeuralNet cnn = (NeuralNet) in.readObject();
            in.close();
            return cnn;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
}
