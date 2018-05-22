package com.strangerws.ssu.edu.textanalyzer.neuralnet.api;

public class Size {

    public final int x;
    public final int y;

    public Size(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public String toString() {
        return "Size(" + " x = " + x + " y= " + y + ")";
    }

    public Size divide(Size scaleSize) {
        int x = this.x / scaleSize.x;
        int y = this.y / scaleSize.y;
        if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
            throw new RuntimeException(this + "" + scaleSize);
        return new Size(x, y);
    }

    public Size subtract(Size size, int append) {
        int x = this.x - size.x + append;
        int y = this.y - size.y + append;
        return new Size(x, y);
    }
}
