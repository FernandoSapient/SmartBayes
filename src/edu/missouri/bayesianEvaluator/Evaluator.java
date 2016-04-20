/**
 * 
 */
package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Allows evaluating a bayesian network with a dataset
 *
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.02 2016-04-19
 * @since {@code bayesianEvaluator} version 0.10 2016-04-19
 */
// TODO: create non-static versions of all methods
// (store the bayesian network you're training as an attribute)
public class Evaluator {

	/**
	 * Creates the specified number of spits of the data. The data is randomized
	 * and split into two parts, each ordering unique. All the "Key" parts are
	 * guaranteed to be the same size amongst themselves. All the "Value" parts
	 * are guaranteed to be the same size amongst themselves. This particularly
	 * useful for several train/test evaluations
	 * 
	 * @param data
	 *            The dataset to be split
	 * @param ratio
	 *            The proportion of the data that each split will have as its
	 * @param number
	 *            The number of splits to perform
	 * @return A {@code Map} with {@code number} entries where each key <i>k</i>
	 *         is an {@code Instances} object containing {@code ratio} of
	 *         {@code data}, and {@code get(k) returns the remaining 1 &minus;
	 *         {@code ratio} of {@code data}
	 */
	public static Map<Instances, Instances> randomSplit(Instances data,
			float ratio, int number) {
		Map<Instances, Instances> out = new HashMap<Instances, Instances>(
				number);
		Random R = new Random();

		int trainSize = Math.round(data.numInstances() * ratio);
		int testSize = data.numInstances() - trainSize;

		for (; number > 0; number--) {
			data.randomize(R);
			out.put(new Instances(data, 0, trainSize), new Instances(data,
					trainSize, testSize));
		}

		return out;
	}

	/**
	 * Takes a (trained?) Bayesian network (provided in an XML BIF file) and evaluates
	 * it using the given data. 
	 * <!--The following is copied from Trainer.main-->
	 * The data file must be on one of Weka's accepted file formats (ARFF,
	 * C4.5, CSV, JSON, LibSVM, MatLab, DAT, BSI, or XRFF, as of Weka version
	 * 3.7.12). The program might fail if the Bayesian network is not in Weka
	 * format (use {@link BifUpdate} for this purpose). A filtering criterion
	 * can be specified, against which only the elements of the first attribute
	 * which are equal to it will be selected.
	 * <p/>
	 * The method discretizes each attribute to match the number of possible values in
	 * its corresponding node (see {@link #discretizeToBayes(Instances, BayesNet, boolean)})
	 * unless the attribute is already discrete.
	 * <p/>
	 * If the file is CSV file, the data is assumed to have originated
	 * from the World Bank, and the missing value placeholder is set to "..",
	 * but this may change in a future version.
	 * 
	 * @param args
	 *            An array containing, at {@code args[0]}, the pat of the data
	 *            file to train the Bayesian network with; at {@code args[1]}
	 *            the path of the file containing the Bayesian network to train;
	 *            at at {@code args[2]} the path of the output file in which to
	 *            store the trained network; and optionally, at {@code args[3]}
	 *            a filtering criterion and at {@code args[4],.
	 * @throws Exception
	 *             If any of the files could not be read
	 * @since {@code bayesianEvaluator} 0.01 2016-04-10
	 */
	// TODO improve parameter handling. See JCLAP for potential solution
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion] [UseFrequencyDiscretization]");
			return;
		}

		DataSource source = new DataSource(args[0]);
		if (source.getLoader() instanceof CSVLoader) {
			((CSVLoader) source.getLoader()).setMissingValue(".."); // This is
																	// bad
																	// practice
		}
		Instances data = source.getDataSet();
		EditableBayesNet bn = BifUpdate.loadBayesNet(args[1]);

		// filter out by criterion
		if (args.length >= 4) {
			System.out.println("Filtering by " + data.attribute(0).name()
					+ " equal to " + args[3] + "...");
			data = Trainer.filterByCriterion(args[3], data, 0);
		}

		System.out.println("Conforming data to network...");
		if (args.length >= 5)
			data = Trainer.conformToNetwork(data, bn,
					Boolean.getBoolean(args[4]));
		else
			data = Trainer.conformToNetwork(data, bn, false);

		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("Setting " + data.classAttribute().name()
				+ " as class...");
		
		float trainSize = 0.9f;

		//TODO move to function
		assert trainSize>0;
		assert trainSize<1;
		
		Map<Instances, Instances> Trainset = randomSplit(data, trainSize, Math.round(1/(1-trainSize)));
		Iterator<Instances> trains = Trainset.keySet().iterator();
		while(trains.hasNext()){
			Instances current = trains.next();
			bn.setData(current);
			bn.estimateCPTs();
			Evaluation e = new Evaluation(current);
			e.evaluateModel(bn, Trainset.get(current));
			System.out.println("\nRESULTS:\n"+e.toSummaryString());
		}
	}

}
