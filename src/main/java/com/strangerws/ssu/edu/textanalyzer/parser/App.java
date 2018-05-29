package com.strangerws.ssu.edu.textanalyzer.parser;


import javacnn.cnn.CNN;
import javacnn.cnn.Layer;
import javacnn.dataset.Dataset;
import javacnn.util.ConcurenceRunner;
import javacnn.util.Log;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_CUBIC;
import static org.bytedeco.javacpp.opencv_imgproc.resize;


public class App {
    private final static String IMAGEPATH = "src\\main\\resources\\sample\\font-test-001.jpg";
    private final static int SIZE = 28;

    public static BufferedImage IplImageToBufferedImage(IplImage src) {
        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        Frame frame = grabberConverter.convert(src);
        return paintConverter.getBufferedImage(frame, 1);
    }

    public static void main(String[] args) throws IOException {
//        App app = new App();
//
//        /*Load image in grayscale mode*/
//        IplImage image = cvLoadImage(IMAGEPATH, 0);
//        imshow("src\\main\\resources\\output\\1 - gray.jpg", new Mat(image)); // Save gray version of image*/
//
//        /*Binarising Image*/
//        IplImage binimg = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
//        cvThreshold(image, binimg, 0, 255, CV_THRESH_OTSU);
//        imshow("src\\main\\resources\\output\\2 - binary.jpg", new Mat(binimg)); // Save binarised version of image*/
//
//        /*Invert image */
//        Mat inverted = new Mat();
//        bitwise_not(new Mat(binimg), inverted);
//        IplImage inverimg = new IplImage(inverted);
//        imshow("src\\main\\resources\\output\\3 - inverted.jpg", new Mat(inverimg)); // Save dilated version of image*/
//
//
//        /*Dilate image to increase the thickness of each digit*/
//        IplImage dilated = cvCreateImage(cvGetSize(inverimg), IPL_DEPTH_8U, 1);
//        opencv_imgproc.cvDilate(inverimg, dilated, null, 1);
//        imshow("src\\main\\resources\\output\\4 - dilated.jpg", new Mat(dilated)); // Save dilated version of image*/
//
//        /*Find countour */
//        CvMemStorage storage = cvCreateMemStorage(0);
//        CvSeq contours = new CvSeq();
//        cvFindContours(dilated.clone(), storage, contours, Loader.sizeof(CvContour.class), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
//        CvSeq ptr;
//        List<Rect> rects = new ArrayList<>();
//        for (ptr = contours; ptr != null; ptr = ptr.h_next()) {
//            CvRect boundbox = cvBoundingRect(ptr, 1);
//            Rect rect = new Rect(boundbox.x(), boundbox.y(), boundbox.width(), boundbox.height());
//            rects.add(rect);
//            cvRectangle(image, cvPoint(boundbox.x(), boundbox.y()),
//                    cvPoint(boundbox.x() + boundbox.width(), boundbox.y() + boundbox.height()),
//                    CV_RGB(0, 0, 0), 1, 0, 0);
//        }
//
//        Mat result = new Mat(image);
//        Collections.sort(rects, new RectComparator());
//
//        List<byte[]> letters = new ArrayList<>();
//
//        for (int i = 0; i < rects.size(); i++) {
//            Rect rect = rects.get(i);
//            Mat letter = new Mat(dilated).apply(rect);
//            //copyMakeBorder(letter, letter, 10, 10, 10, 10, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
//            resize(letter, letter, new Size(28, 28), 0, 0, INTER_CUBIC);
//            imwrite("src\\main\\resources\\output\\letter-" + (i + 1) + "-.jpg", letter);
//
//            byte tmp[] = new byte[letter.cols() * letter.rows()];
//            letter.data().get(tmp);
//            letters.add(tmp);
//
//            //TODO
//            //Set input image to mat
//            //Train, Predict, Test refactor
//            //Train net by mat images
//            //Send letters by one and print it
//        }

        runNet();
//        imwrite("src\\main\\resources\\output\\result.jpg", result);//save final result
    }

    public static List<double[]> getTrainingData() {
        List<double[]> result = new ArrayList<>();
        String pathBegin = "src\\main\\resources\\trainingData\\";
        StringBuilder path;

        for (int i = 1; i < 80; i++) {
            for (int j = 1; j <= 30; j++) {
                path = new StringBuilder(pathBegin)
                        .append(i)
                        .append("\\")
                        .append(j)
                        .append(".png");
                byte[] tmp = new byte[SIZE * SIZE];
                Mat mat = new Mat(cvLoadImage(path.toString(), CV_LOAD_IMAGE_GRAYSCALE));
                resize(mat, mat, new opencv_core.Size(SIZE, SIZE), 0, 0, INTER_CUBIC);
                mat.data().get(tmp);
                result.add(doubleCast(tmp));
            }
        }

        return result;
    }

    private static double[] doubleCast(byte[] tmp) {
        double[] casted = new double[tmp.length];
        for (int i = 0; i < tmp.length; i++) {
            casted[i] = tmp[i] < 0 ? 0 : 1;
        }
        return casted;
    }

    public static List<Integer> getTrainingAnswers() {
        List<Integer> result = new ArrayList<>();

        for (int i = 1; i < 80; i++) {
            for (int j = 0; j < 30; j++) {
                result.add(i);
            }
        }

        return result;
    }

    private static void runNet() {
        Dataset dataset = new Dataset();

        final ConcurenceRunner concurenceRunner = new ConcurenceRunner();
        CNN.LayerBuilder builder = new CNN.LayerBuilder();
        builder.addLayer(Layer.buildInputLayer(new javacnn.cnn.Layer.Size(SIZE, SIZE)));
        builder.addLayer(Layer.buildConvLayer(6, new javacnn.cnn.Layer.Size(5, 5)));
        builder.addLayer(Layer.buildSampLayer(new javacnn.cnn.Layer.Size(2, 2)));
        builder.addLayer(Layer.buildConvLayer(12, new javacnn.cnn.Layer.Size(5, 5)));
        builder.addLayer(Layer.buildSampLayer(new javacnn.cnn.Layer.Size(2, 2)));
        builder.addLayer(Layer.buildOutputLayer(27));
        CNN cnn = new CNN(builder, 50, concurenceRunner);
        setDataset(dataset);
        Log.switchOn();
        cnn.train(dataset, 100);

    }

    private static void imshow(String mock, Mat img) {
        CanvasFrame canvasFrame = new CanvasFrame("Preview");
        canvasFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        canvasFrame.setCanvasSize(img.cols(), img.rows());
        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(img));
    }


    private static void setDataset(Dataset dataset) {
        String pathBegin = "src\\main\\resources\\trainingData\\";
        StringBuilder path;
        for (int l = 0; l < 10; l++) {
            for (int i = 1; i <= 26; i++) {
                for (int j = 1; j <= 30; j++) {
                    path = new StringBuilder(pathBegin)
                            .append(i)
                            .append("\\")
                            .append(j)
                            .append(".png");
                    byte[] tmp = new byte[SIZE * SIZE];
                    Mat mat = new Mat(cvLoadImage(path.toString(), CV_LOAD_IMAGE_GRAYSCALE));
                    resize(mat, mat, new opencv_core.Size(SIZE, SIZE), 0, 0, INTER_CUBIC);
                    mat.data().get(tmp);
                    dataset.append(doubleCast(tmp), (double) i);
                }
            }
        }

        dataset.shuffle();
    }

}
