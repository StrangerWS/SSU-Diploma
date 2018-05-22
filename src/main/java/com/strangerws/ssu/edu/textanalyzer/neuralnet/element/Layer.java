package com.strangerws.ssu.edu.textanalyzer.neuralnet.element;

import com.strangerws.ssu.edu.textanalyzer.neuralnet.api.LayerType;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size;
import com.strangerws.ssu.edu.textanalyzer.util.Utils;

/**
 * Класс, создающий слои для нейросети. Может создавать входные, выходные, свёрточные и обычные слои
 *
 * @author StrangerWS (Artem Dobrynin)
 * @see com.strangerws.ssu.edu.textanalyzer.neuralnet.NeuralNet
 * @see LayerType
 * @see Size
 */
public class Layer {
    /**
     * Количество записанных итераций по текущей выборке
     */
    private static int writeBatch = 0;

    /**
     * Тип текущего слоя
     *
     * @see LayerType
     */
    private LayerType type;

    /**
     * Размер текущего слоя
     *
     * @see Size
     */
    private Size size;
    /**
     * Размер ядра слоя
     *
     * @see Size
     */
    private Size kernelSize;
    /**
     * Размер масштабирующего слоя
     *
     * @see Size
     */
    private Size scaleSize;

    private int outputCount;

    private double[][][][] kernel;
    private double[] bias;
    private double[][][][] outputMaps;
    private double[][][][] errors;

    private int categoriesNumber = -1;

    public LayerType getType() {
        return type;
    }

    public Size getSize() {
        return size;
    }

    public Size getKernelSize() {
        return kernelSize;
    }

    public Size getScaleSize() {
        return scaleSize;
    }

    public int getOutputCount() {
        return outputCount;
    }

    public double[] getBias() {
        return bias;
    }

    public double[][][][] getOutputMaps() {
        return outputMaps;
    }

    public int getCategoriesNumber() {
        return categoriesNumber;
    }

    public void setSize(Size size) {
        this.size = size;
    }

    public void setOutputCount(int outputCount) {
        this.outputCount = outputCount;
    }

    private Layer() {
    }

    /**
     * Сбрасывает значение итераций по выборке
     */
    public static void resetBatch() {
        writeBatch = 0;
    }

    /**
     * Используется для подсчёта количества итераций по выборке
     */
    public static void incrementBatch() {
        writeBatch++;
    }

    /**
     * Генерация входного слоя
     *
     * @param mapSize размер входного слоя
     * @return Входной слой нейросети
     */
    public static Layer buildInputLayer(Size mapSize) {
        Layer layer = new Layer();
        layer.type = LayerType.INPUT;
        layer.outputCount = 1;
        layer.size = mapSize;
        return layer;
    }

    /**
     * Генерация свёрточного слоя
     *
     * @param outputCount количество выходных слоёв
     * @param kernelSize  размер ядра
     * @return свёрточный слой
     */
    public static Layer buildConvolutionalLayer(int outputCount, Size kernelSize) {
        Layer layer = new Layer();
        layer.type = LayerType.CONVOLUTIONAL;
        layer.outputCount = outputCount;
        layer.kernelSize = kernelSize;
        return layer;
    }

    /**
     * Генерация слоя подвыборки
     *
     * @param scaleSize множитель слоя
     * @return слой подвыборки
     */
    public static Layer buildSampleLayer(Size scaleSize) {
        Layer layer = new Layer();
        layer.type = LayerType.SAMPLE;
        layer.scaleSize = scaleSize;
        return layer;
    }

    /**
     * Генерация выходного слоя, определение количества выходных нейронов относительно количества категорий
     *
     * @param categoriesNumber количество категорий, которые нейросеть может распознать
     * @return выходной слой нейросети
     */
    public static Layer buildOutputLayer(int categoriesNumber) {
        Layer layer = new Layer();
        layer.categoriesNumber = categoriesNumber;
        layer.type = LayerType.OUTPUT;
        layer.size = new Size(1, 1);
        layer.outputCount = categoriesNumber;
        return layer;
    }


    /**
     * Заполнение свёрточного ядра случайными значениями
     *
     * @param frontMapNum
     */
    public void initKernel(int frontMapNum) {
        this.kernel = new double[frontMapNum][outputCount][kernelSize.x][kernelSize.y];
        for (int i = 0; i < frontMapNum; i++)
            for (int j = 0; j < outputCount; j++)
                kernel[i][j] = Utils.randomMatrix(kernelSize.x, kernelSize.y);
    }

    /**
     * Заполнение ядра выходного свёрточного слоя. Размерность ядра равна размерности предыдущего слоя
     *
     * @param frontMapNum
     * @param size размер ядра
     */
    public void initOutputKernel(int frontMapNum, Size size) {
        kernelSize = size;
        this.kernel = new double[frontMapNum][outputCount][kernelSize.x][kernelSize.y];
        for (int i = 0; i < frontMapNum; i++)
            for (int j = 0; j < outputCount; j++)
                kernel[i][j] = Utils.randomMatrix(kernelSize.x, kernelSize.y);
    }

    /**
     * Инициализация смещения
     */
    public void initBias() {
        this.bias = Utils.randomArray(outputCount);
    }

    /**
     * Инициализация выходного слоя
     *
     * @param batchSize размер выборки
     */
    public void initOutputMaps(int batchSize) {
        outputMaps = new double[batchSize][outputCount][size.x][size.y];
    }

    /**
     * Устанавливает значение текущего слоя
     *
     * @param mapNo номер выборки
     * @param mapX  столбец матрицы
     * @param mapY  строка матрицы
     * @param value устанавливаемое значение
     */
    public void setMapValue(int mapNo, int mapX, int mapY, double value) {
        outputMaps[writeBatch][mapNo][mapX][mapY] = value;
    }

    /**
     * Устанавливает готовую матрицу в определённой выборке
     *
     * @param mapNo номер выборки
     * @param outMatrix устанавливаемая матрица
     */
    public void setMapValue(int mapNo, double[][] outMatrix) {
        outputMaps[writeBatch][mapNo] = outMatrix;
    }

    /**
     * Get the index map matrix. In performance considerations, no duplicate objects are returned, but the reference is returned directly, and the caller must be careful.
     * Avoid modifying outmaps. If you need to modify, please call setMapValue(...)
     *
     * @param index
     * @return
     */
    public double[][] getMap(int index) {
        return outputMaps[writeBatch][index];
    }

    /**
     * Get the convolution kernel of the previous layer i map to the current layer j map
     *
     * @param i Uplevel map subscript
     * @param j Current level of map subscript
     * @return
     */
    public double[][] getKernel(int i, int j) {
            return kernel[i][j];

}

    /**
     * Set residual value
     *
     * @param mapNo
     * @param mapX
     * @param mapY
     * @param value
     */
    public void setError(int mapNo, int mapX, int mapY, double value) {
        errors[writeBatch][mapNo][mapX][mapY] = value;
    }

    /**
     * Set the residual value as a map matrix block
     *
     * @param mapNo
     * @param matrix
     */
    public void setError(int mapNo, double[][] matrix) {
        // Log.i(type.toString());
        // Util.printMatrix(matrix);
        errors[writeBatch][mapNo] = matrix;
    }

    /**
     * Get the mapNo map of the residual. Does not return a copy of the object, but directly returns a reference, the caller should be cautious,
     * Avoid modifying errors. If you need to modify, please call setError(...)
     *
     * @param mapNo
     * @return
     */
    public double[][] getError(int mapNo) {
        return errors[writeBatch][mapNo];
    }

    /**
     * Get all the residuals (each record and each map)
     *
     * @return
     */
    public double[][][][] getErrors() {
        return errors;
    }

    /**
     * Initialize the residual array
     *
     * @param batchSize
     */
    public void initErros(int batchSize) {
        errors = new double[batchSize][outputCount][size.x][size.y];
    }

    /**
     * @param lastMapNo
     * @param mapNo
     * @param kernel
     */
    public void setKernel(int lastMapNo, int mapNo, double[][] kernel) {
        this.kernel[lastMapNo][mapNo] = kernel;
    }

    /**
     * Get the mapNo number
     *
     * @param mapNo
     * @return
     */
    public double getBias(int mapNo) {
        return bias[mapNo];
    }

    /**
     * Set the offset value of the mapNo map
     *
     * @param mapNo
     * @param value
     */
    public void setBias(int mapNo, double value) {
        bias[mapNo] = value;
    }

    /**
     * Get each map of the batch matrix
     *
     * @return
     */

    public double[][][][] getMaps() {
        return outputMaps;
    }

    /**
     * Get the residual of mapNo in the recordId record
     *
     * @param recordId
     * @param mapNo
     * @return
     */
    public double[][] getError(int recordId, int mapNo) {
        return errors[recordId][mapNo];
    }

    /**
     * Get the output mapmapNo recordId record
     *
     * @param recordId
     * @param mapNo
     * @return
     */
    public double[][] getMap(int recordId, int mapNo) {
        return outputMaps[recordId][mapNo];
    }

    /**
     * Get the number of categories
     *
     * @return
     */
    public int getClassNum() {
        return categoriesNumber;
    }

    /**
     * Get all convolution kernels
     *
     * @return
     */
    public double[][][][] getKernel() {
        return kernel;
    }
}
