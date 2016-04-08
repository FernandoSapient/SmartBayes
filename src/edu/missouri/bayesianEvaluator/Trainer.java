package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.SubsetByExpression;
import weka.filters.unsupervised.attribute.Discretize;
/**
 * Contains methods to train (i.e. compute the Conditional Probability Tables)
 * of a Bayesian Network.
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.04 2016-04-06
 * @since {@link bayesianEvaluator} version 0.02 2016-04-02
 */
public class Trainer {
	/**Gets the names of all the nodes in the given bayesian network
	 * 
	 * @param bn The bayesian Network to get the node names from
	 * @return A {@code Set} containing all of the node names
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
	 * A filtering criterion can be specified, against which only the elements
	 * of the first attribute which are equal to it will be selected.
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
	// TODO figure out a way to parse any regular expression as filter criterion
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion] [UseFrequencyDiscretization]");
			return;
		}

		DataSource source = new DataSource(args[0]);
		((CSVLoader)source.getLoader()).setMissingValue("..");	//This is bad practice
			//TODO: Move everything to a method receiving an Instances object to force the caller to set his own missing values 
		assert ((CSVLoader)source.getLoader()).getMissingValue().equals("..");	//checking if it was done correctly
		Instances data = source.getDataSet();
		//TODO check if all are numeric and decide what to do if they aren't
		BayesNet bn = BifUpdate.loadBayesNet(args[1]);
		Set<String> nodes = getNodeNames(bn);

		//filter out by criterion
		if (args.length >= 4) {
			System.out.println("Filtering by " + data.attribute(0).name() + " equal to "
					+ args[3] + "...");
			// TODO move to method so garbage collector can take care of it
			data.setClassIndex(0);
			SubsetByExpression f = new SubsetByExpression();
			f.setInputFormat(data);
			f.setExpression("CLASS is \'" + args[3] + "\'");
			data = Filter.useFilter(data, f);
			//TODO assert containsOnly(data.attribute(0), args[3])
		}

		
		// remove attributes not in the bayesian network
		int attributes = data.numAttributes();
		Vector<String> attributeNames = new Vector<String>(nodes.size());
		int checked = 0;
		Remove r = new Remove();
		
		for (int i = 0; i<data.numAttributes();) {
			String a = data.attribute(i).name();
			if (!nodes.contains(a)) {
				r.setAttributeIndices(Integer.toString(i+1));
				r.setInputFormat(data);
				data = Filter.useFilter(data, r);
				assert data.attribute(a) == null; // it must have actually been
													// removed
			}else{
				attributeNames.add(a);
				i++;
			}
			
			checked++;
		}
		assert checked == attributes;
		assert data.numAttributes() == nodes.size();
		assert nodes.containsAll(attributeNames);
		assert attributeNames.containsAll(nodes);
		
		//discretize data according to BN value labels
		Discretize d = new Discretize();
		//d.setIgnoreClass(true);	//not needed, class is already unset
		for(int i = 0; i<nodes.size(); i++){
			int j = attributeNames.indexOf(bn.getNodeName(i));
			if(j == -1)
				throw new IllegalArgumentException("The Bayesian network in "+args[1]
						+" contains node "+bn.getNodeName(i)+" but no such node "
						+"exists in the data file "+args[0]);
			d.setAttributeIndices(Integer.toString(j+1));
			d.setBins(bn.getCardinality(i));
			if (args.length >= 5)
				d.setUseEqualFrequency(Boolean.getBoolean(args[4]));
			d.setInputFormat(data);
			data = Filter.useFilter(data, d);
			assert bn.getNodeName(i).equals(data.attribute(j).name());
			assert data.numDistinctValues(j) == bn.getCardinality(i); 
		}
			
		
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println(data.classAttribute().name()+" set as class");

		bn.m_Instances = data;
		bn.estimateCPTs();	//TODO: Find out why this fails

		PrintWriter f = new PrintWriter(args[2]);
		f.write(bn.toXMLBIF03());
		f.close();
		System.out.println(args[2]+" created successfully");
	}
}
