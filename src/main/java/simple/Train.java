/**
 * Developer: Minhas Kamal (BSSE: 0509, IID, DU)
 * Date: 09.Sep.2014
**/

package simple;

import com.strangerws.ssu.edu.textanalyzer.util.Matrix;
import simple.weightedPixel.WeightedStandardImage;
import simple.weightedPixel.WeightedStandardPixelTrainer;

import java.io.File;
import java.util.ArrayList;



public class Train {
	public static void main(String[] args) throws Exception {

		//list of image files//////////////////////////////////////////////////////////////////////////////////////////
		int types = 79;
		String[] stringFilePaths = new String[types];
		for(int i=1; i<=types; i++){
			stringFilePaths[i-1] = "src/main/resources/trainingData/" + i;
		}

		ArrayList<String> filePathList = new ArrayList<String>();
		ArrayList<Integer> idList = new ArrayList<Integer>();

		int id=1;
		for(String stringFilePath: stringFilePaths){

			File[] files = new File(stringFilePath).listFiles();

			for(File file: files){
				filePathList.add(file.getAbsolutePath());
				idList.add(id);
			}

			id++;
		}

		String[] filePaths = new String[filePathList.size()];
		filePathList.toArray(filePaths);
		Integer[] ids = new Integer[idList.size()];
		idList.toArray(ids);


		///test
		/*for(int i=filePaths.length-1; i>=0; i--){
			System.out.println("filePaths: " + filePaths[i]);
			System.out.println("ids: " + ids[i]);
		}*/


		//train/////////////////////////////////////////////////////////////////////////////////////////////////////////
		WeightedStandardPixelTrainer weightedStandardPixelTrainer = new WeightedStandardPixelTrainer(new int[]{45, 45});

		weightedStandardPixelTrainer.train(filePaths, ids);
		WeightedStandardImage weightedStandardImage = weightedStandardPixelTrainer.getWeightedStandardImage();
		weightedStandardImage.saveKnowledge("src/main/resources/knowledge/KnowledgeAlphabet.log");

		///test
		for(int i=0; i<types; i++){
			Matrix mat = weightedStandardImage.getStandardImages(i);
			mat.write("src/main/resources/knowledge/imageRepresentation/stdImage " + i + ".png");
		}

		System.out.println("Operation Successful!!!");
	}
}
