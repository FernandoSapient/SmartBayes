package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;

import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.SubsetByExpression;

/**
 * Contains methods to train (i.e. compute the Conditional Probability Tables)
 * of a Bayesian Network.
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.02 2016-04-03
 * @since {@link bayesianEvaluator} version 0.02 2016-04-02
 */
public class Trainer {
	/**
	 * 
	 * @param bn
	 * @return
	 */
	private static Set<String> getNodeNames(BayesNet bn) {
		Set<String> out = new HashSet<String>();
		for (int i = 0; i < bn.getNrOfNodes(); i++)
			out.add(bn.getNodeName(i));
		return out;
	}

	/**
	 * Trains a Bayesian network (provided in an XML BIF file) using the given
	 * data. The data file must be on one of Weka's accepted file formats (ARFF,
	 * C4.5, CSV, JSON, LibSVM, MatLab, DAT, BSI, or XRFF, as of Weka version
	 * 3.7.12). The program might fail if the Bayesian network is not in Weka
	 * format (use {@link BifUpdate} for this purpose).
	 * 
	 * @param args
	 *            An array containing, at {@code args[0]}, the pat of the data
	 *            file to train the Bayesian network with; at {@code args[1]}
	 *            the path of the file containing the Bayesian network to train;
	 *            at at {@code args[2]} the path of the output file in which to
	 *            store the trained network; and optionally, at {@code args[3]}
	 *            a filtering criterion.
	 * @throws Exception
	 *            If any of the files could not be read
	 */
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion]");
			return;
		}
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		BayesNet bn = BifUpdate.loadBayesNet(args[1]);

		if (args.length >= 3) {
			// TODO move to method so garbage collector can take care of it
			// TODO figure out a way to parse any regular expression in this
			// parameter
			data.setClassIndex(0);
			SubsetByExpression f = new SubsetByExpression();
			f.setInputFormat(data);
			f.setExpression("CLASS is \'" + args[3] + "\'");
			data = Filter.useFilter(data, f);
		}
		data.setClassIndex(data.numAttributes() - 1);

		bn.m_Instances = data;
		bn.estimateCPTs();

		PrintWriter f = new PrintWriter(args[2]);
		f.write(bn.toXMLBIF03());
		f.close();
	}
}
