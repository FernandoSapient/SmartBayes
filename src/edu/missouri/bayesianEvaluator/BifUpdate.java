/**Converts the given bif file into the Weka format by loading it into Weka and then writing it back.
 * Usage: {@code java Main <input BIF file> <output BIF file>}
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.02 2016-03-29
 * @since {@link bayesianEvaluator} version 0.02 2016-04-02
 */
package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.EditableBayesNet;

/**Provides methods for loading and saving Bif Files in into the Weka format 

 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version {@link bayesianEvaluator} version 0.02 2016-04-02
 * @since {@link bayesianEvaluator} version 0.01 2016-04-01
 */
public class BifUpdate {
	
	/**Loads a Bayesian network from an XML BIF file into a weka {@code BayesNet}
	 * 
	 * @param filename The path of an XML BIF file to open
	 * @return a {@code weka.classifiers.bayes.BayesNet} object with the structure in the file
	 * @throws Exception If the file could not be read correctly
	 * @since version 0.02 2016-04-02
	 */
	public static BayesNet loadBayesNet(String filename) throws Exception{
		BIFReader br = new BIFReader();
		br.processFile(filename);
		return new EditableBayesNet(br);
	}
	
	/**Converts the given bif file into the Weka format by loading it into Weka and then writing it back.
	 * The program may print warnings and non-fatal errors if the input file is not already in Weka format.
	 *  
	 * 
	 * @param args an array containing the input ({@code args[0]}) and output ({@code args[1]}) filenames
	 * @throws Exception if file reading or writing fails.
	 * @since version 0.01 2016-04-01
	 */
	public static void main(String[] args) throws Exception{
		if (args.length < 1) {
			System.err.println("Usage: java BifUpdate <input filename> <output filename>");
			return;
		}

		BayesNet bn = loadBayesNet(args[0]);
		//bn.updateClassifier(data);
		//bn.estimateCPTs();
		PrintWriter f = new PrintWriter(args[1]);
		f.write( bn.toXMLBIF03() );
		f.close();
	}

}
