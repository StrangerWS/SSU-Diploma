package com.strangerws.ssu.edu.textanalyzer.parser;

import com.strangerws.ssu.edu.textanalyzer.neuralnet.LayerBuilder;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.NeuralNet;
import com.strangerws.ssu.edu.textanalyzer.neuralnet.element.Layer;
import com.strangerws.ssu.edu.textanalyzer.util.Dataset;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import com.strangerws.ssu.edu.textanalyzer.util.RectComparator;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.javacpp.helper.opencv_core.CV_RGB;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgproc.*;


public class App {
    private final static String IMAGEPATH = "C:\\Users\\Artem_Dobrynin\\IdeaProjects\\TextAnalyzer\\src\\main\\resources\\sample\\test.png";

    public static BufferedImage IplImageToBufferedImage(IplImage src) {
        OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter paintConverter = new Java2DFrameConverter();
        Frame frame = grabberConverter.convert(src);
        return paintConverter.getBufferedImage(frame, 1);
    }

    public static void displayImage(Mat imgage) {
        BufferedImage img = IplImageToBufferedImage(new IplImage(imgage));
        JFrame frame = new JFrame();
        frame.setTitle("Result");
        frame.setSize(new Dimension(img.getWidth() + 10, img.getHeight() + 10));
        frame.setLayout(new FlowLayout());
        JLabel label = new JLabel();
        label.setIcon(new ImageIcon(img));
        frame.add(label);
        frame.setResizable(false);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }



    public static void main(String[] args) throws IOException {
        App app = new App();

        /*Load image in grayscale mode*/
        IplImage image = cvLoadImage(IMAGEPATH, 0);
        imshow(new Mat(image)); // Save gray version of image*/

        /*Binarising Image*/
        IplImage binimg = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
        cvThreshold(image, binimg, 0, 255, CV_THRESH_OTSU);
        imshow(new Mat(binimg)); // Save binarised version of image*/

        /*Invert image */
        Mat inverted = new Mat();
        bitwise_not(new Mat(binimg), inverted);
        IplImage inverimg = new IplImage(inverted);
        imshow(new Mat(inverimg)); // Save dilated version of image*/


        /*Dilate image to increase the thickness of each digit*/
        IplImage dilated = cvCreateImage(cvGetSize(inverimg), IPL_DEPTH_8U, 1);
        opencv_imgproc.cvDilate(inverimg, dilated, null, 1);
        imshow(new Mat(dilated)); // Save dilated version of image*/

        /*Find countour */
        CvMemStorage storage = cvCreateMemStorage(0);
        CvSeq contours = new CvSeq();
        cvFindContours(dilated.clone(), storage, contours, Loader.sizeof(CvContour.class), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
        CvSeq ptr;
        List<Rect> rects = new ArrayList<>();
        for (ptr = contours; ptr != null; ptr = ptr.h_next()) {
            CvRect boundbox = cvBoundingRect(ptr, 1);
            Rect rect = new Rect(boundbox.x(), boundbox.y(), boundbox.width(), boundbox.height());
            rects.add(rect);
            cvRectangle(image, cvPoint(boundbox.x(), boundbox.y()),
                    cvPoint(boundbox.x() + boundbox.width(), boundbox.y() + boundbox.height()),
                    CV_RGB(0, 0, 0), 2, 0, 0);
        }

        Mat result = new Mat(image);
        Collections.sort(rects, new RectComparator());

        List<byte[]> letters = new ArrayList<>();

        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            Mat letter = new Mat(dilated).apply(rect);
            //imshow(letter);
            //copyMakeBorder(letter, letter, 10, 10, 10, 10, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
            resize(letter, letter, new Size(28, 28), 0, 0, INTER_CUBIC);


            byte tmp[] = new byte[letter.cols()*letter.rows()];
            letter.data().get(tmp);
            System.out.println(tmp.length);
            letters.add(tmp);

            //TODO
            //Set input image to mat
            //Train net by mat images
            //Send letters by one and print it
        }


        Dataset<byte[], Character> dataset = new Dataset<>();
        dataset.setDataset(letters, Arrays.asList('H','a','l','l','o','w','!','.','I','a','m','M','i','.','n','h','a','s','K','a','m','a','l','!','.'));

        LayerBuilder builder = new LayerBuilder();
        builder
                .addLayer(Layer.buildInputLayer(new com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size(28, 28)))
                .addLayer(Layer.buildConvolutionalLayer(6, new com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size(5, 5)))
                .addLayer(Layer.buildSampleLayer(new com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size(2,2)))
                .addLayer(Layer.buildConvolutionalLayer(12, new com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size(5,5)))
                .addLayer(Layer.buildSampleLayer(new com.strangerws.ssu.edu.textanalyzer.neuralnet.api.Size(2, 2)))
                .addLayer(Layer.buildOutputLayer(10));
        NeuralNet cnn = new NeuralNet(builder, 50);

        imshow(result);//save final result
    }

    private static void imshow(Mat img) {
        CanvasFrame canvasFrame = new CanvasFrame("Preview");
        canvasFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        canvasFrame.setCanvasSize(img.cols(), img.rows());
        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(img));
    }


}
