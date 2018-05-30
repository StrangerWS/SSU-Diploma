package com.strangerws.ssu.edu.textanalyzer.util;

import java.util.Comparator;

import org.bytedeco.javacpp.opencv_core;

public class RectComparator implements Comparator<opencv_core.Rect> {

    @Override
    public int compare(opencv_core.Rect t1, opencv_core.Rect t2) {

        if (Integer.compare(t1.y(), t2.y()) == 0) {
            return Integer.compare(t1.x(), t2.x());
        } else return Integer.compare(t1.y(), t2.y());
    }
}