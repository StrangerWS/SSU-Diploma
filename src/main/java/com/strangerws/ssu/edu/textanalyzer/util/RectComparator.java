package com.strangerws.ssu.edu.textanalyzer.util;

import java.util.Comparator;
import org.bytedeco.javacpp.opencv_core;

public class RectComparator implements Comparator<opencv_core.Rect> {

    @Override
    public int compare(opencv_core.Rect t1, opencv_core.Rect t2) {
        return Integer.valueOf(t1.y()).compareTo(t2.y());
    }
}