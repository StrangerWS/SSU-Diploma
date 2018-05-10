import parser.Image;
import parser.Threshold;

public class Main {
    public static void main(String[] args) {
        Image image = new Image();
        image.readImage("C:\\Users\\Artem_Dobrynin\\Documents\\Qxv4Qhh_gDY.jpg");
        Threshold.toBinary(image, 50);
        image.writeImage("C:\\Users\\Artem_Dobrynin\\Documents\\threshold.jpg");

    }
}
