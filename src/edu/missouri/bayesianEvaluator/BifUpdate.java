/**Converts the given bif file into the Weka format by loading it into Weka and then writing it back.
Usage: java Main <input BIF file> <output BIF file>
 */
package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.EditableBayesNet;

public class BifUpdate {

	/**Converts the given bif file into the Weka format by loading it into Weka and then writing it back.
	 * The program may print warnings and non-fatal errors if the input file is not already in Weka format.
	 *  
	 * 
	 * @param args an array containing the input ({@code args[0]}) and output ({@code args[1]}) filenames
	 * @throws Exception if file reading or writing fails.
	 */
	public static void main(String[] args) throws Exception{
		if (args.length < 1) {
			System.err
					.println("Usage: java Main <input filename> <output filename>");
			return;
		}

		BIFReader br = new BIFReader();
		br.processFile(args[0]);
		BayesNet bn = new EditableBayesNet(br);
		//bn.updateClassifier(data);
		//bn.estimateCPTs();
		PrintWriter f = new PrintWriter(args[1]);
		f.write( bn.toXMLBIF03() );
		f.close();
	}

}
