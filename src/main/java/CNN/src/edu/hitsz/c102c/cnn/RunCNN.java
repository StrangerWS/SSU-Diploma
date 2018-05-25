package CNN.src.edu.hitsz.c102c.cnn;

import CNN.src.edu.hitsz.c102c.cnn.CNN.LayerBuilder;
import CNN.src.edu.hitsz.c102c.cnn.Layer.Size;
import CNN.src.edu.hitsz.c102c.dataset.Dataset;
import CNN.src.edu.hitsz.c102c.util.ConcurenceRunner;
import CNN.src.edu.hitsz.c102c.util.TimedTest;
import CNN.src.edu.hitsz.c102c.util.TimedTest.TestTask;
import com.strangerws.ssu.edu.textanalyzer.parser.App;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.List;

public class RunCNN {

	public static void runCnn() {
		//����һ�����������
		LayerBuilder builder = new LayerBuilder();
		builder.addLayer(Layer.buildInputLayer(new Size(28, 28)));
		builder.addLayer(Layer.buildConvLayer(6, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildConvLayer(12, new Size(5, 5)));
		builder.addLayer(Layer.buildSampLayer(new Size(2, 2)));
		builder.addLayer(Layer.buildOutputLayer(10));
		CNN cnn = new CNN(builder, 50);

		String fileName = "src/main/resources/dataset/train.format";
		Dataset dataset = Dataset.load(fileName, ",", 784);
		//cnn.train(setDataFromRes(), 5);
		cnn.train(dataset, 3);//
		String modelName = "src/main/resources/model/model.cnn";
		cnn.saveModel(modelName);		
		dataset.clear();
		dataset = null;

		Dataset testset = Dataset.load("src/main/resources/dataset/test.format", ",", -1);
		//cnn.predict(testset, "src/main/resources/dataset/test.predict");
	}

	public static void main(String[] args) {

		new TimedTest(() -> runCnn(), 1).test();
		ConcurenceRunner.stop();

	}

	private static Dataset setDataFromRes(){
		List<double[]> data = App.getTrainingData();
		List<Double> results = App.getTrainingAnswers();
		Dataset dataset = new Dataset();
		List<double[]> doubles = new ArrayList<>();
		for (int i = 0; i < data.size(); i++) {
			double[] tmp = new double[data.get(i).length];
			for (int j = 0; j < data.get(i).length; j++) {
				tmp[j] = data.get(i)[j];
			}
            for (int j = 0; j < 20; j++) {
                dataset.append(tmp, results.get(i));
            }
		}
		dataset.shuffle();


		return dataset;
	}

}
