package edu.missouri.bayesianEvaluator;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Vector;

import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
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
 * @version 0.07 2016-04-09
 * @since {@link bayesianEvaluator} version 0.02 2016-04-02
 */
public class Trainer {
	/**
	 * Gets the names of all the nodes in the given bayesian network
	 * 
	 * @param bn
	 *            The bayesian Network to get the node names from
	 * @return A {@code Set} containing all of the node names
	 * 
	 * @since 0.01 2016-04-02
	 */
	private static List<String> getNodeNames(BayesNet bn) {
		List<String> out = new Vector<String>(bn.getNrOfNodes());
		for (int i = 0; i < bn.getNrOfNodes(); i++)
			out.add(bn.getNodeName(i));
		return out;
	}

	/**
	 * Trains a Bayesian network (provided in an XML BIF file) using the given
	 * data. The data file must be on one of Weka's accepted file formats (ARFF,
	 * C4.5, CSV, JSON, LibSVM, MatLab, DAT, BSI, or XRFF, as of Weka version
	 * 3.7.12). The program might fail if the Bayesian network is not in Weka
	 * format (use {@link BifUpdate} for this purpose). A filtering criterion
	 * can be specified, against which only the elements of the first attribute
	 * which are equal to it will be selected.
	 * 
	 * @param args
	 *            An array containing, at {@code args[0]}, the pat of the data
	 *            file to train the Bayesian network with; at {@code args[1]}
	 *            the path of the file containing the Bayesian network to train;
	 *            at at {@code args[2]} the path of the output file in which to
	 *            store the trained network; and optionally, at {@code args[3]}
	 *            a filtering criterion.
	 * @throws Exception
	 *             If any of the files could not be read
	 * @since 0.01 2016-04-02
	 */
	// TODO figure out a way to parse any regular expression as filter criterion
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion] [UseFrequencyDiscretization]");
			return;
		}

		DataSource source = new DataSource(args[0]);
		((CSVLoader) source.getLoader()).setMissingValue(".."); // This is bad
																// practice
		// TODO: Move everything to a method receiving an Instances object to
		// force the caller to set his own missing values
		assert ((CSVLoader) source.getLoader()).getMissingValue().equals(".."); // checking
																				// if
																				// it
																				// was
																				// done
																				// correctly
		Instances data = source.getDataSet();
		// TODO check if all are numeric and decide what to do if they aren't
		BayesNet bn = BifUpdate.loadBayesNet(args[1]);
		List<String> nodes = getNodeNames(bn);

		// filter out by criterion
		if (args.length >= 4) {
			System.out.println("Filtering by " + data.attribute(0).name()
					+ " equal to " + args[3] + "...");
			data = filterByCriterion(args[3], data, 0);
		}
		
		System.out.println("Conforming data to network...");
		// remove attributes not in the bayesian network
		data = restrictToAttributeSet(data, nodes);

		// reorder to match (required by estimateCPTs)
		data = reorderAttributes(data, nodes);

		// discretize data according to BN value labels
		if (args.length >= 5)
			data = discretizeToBayes(data, bn, Boolean.getBoolean(args[4]));
		else
			data = discretizeToBayes(data, bn, false);

		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("Setting "+data.classAttribute().name() + " as class...");

		bn.m_Instances = data;
		System.out.println("Training network...");
		bn.estimateCPTs();

		PrintWriter f = new PrintWriter(args[2]);
		f.write(bn.toXMLBIF03());
		f.close();
		System.out.println(args[2] + " created successfully");
	}

	/**
	 * Reorders the attributes in the given dataset to be in the order specified
	 * </P> Note: {@code data} <em>will be modified</em> by this function
	 * 
	 * @param data
	 *            The dataset to be ordered
	 * @param attributeOrder
	 *            A list of attribute names giving the order desired
	 * @return A modified dataset with each instance having its attributes in
	 *         the order given by {@code attributeOrder}
	 * @throws IllegalArgumentException
	 *             if {@code attributeOrder} is not composed of the attributes
	 *             from the given dataset
	 * @since 0.07 2016-04-09
	 */
	public static Instances reorderAttributes(Instances data,
			List<String> attributeOrder) throws IllegalArgumentException {
		int n = data.numAttributes();
		if (attributeOrder.size() != n)
			throw new IllegalArgumentException(
					"The number of attributes"
							+ " in the ordering does not match the number of attributes"
							+ " in the data");
		List<String> attributeNames = getAttributeNames(data);
		
		//test "equality" by double contention
		assert attributeOrder.containsAll(attributeNames); // TODO: throw a
															// proper exception
		assert attributeNames.containsAll(attributeOrder); // TODO: throw a
															// proper exception

		int m = data.numInstances();

		// convert each attribute to column vectors and reinsert
		for (int i = 0; i < n; i++) {
			attributeNames = getAttributeNames(data); //the order of the attribute names will change dynamically
			int source = attributeNames.indexOf(attributeOrder.get(i));
			assert source != -1; //if it is, double contention test was wrong
			
			Attribute a = data.attribute(source);
			assert a.name().equals(attributeOrder.get(i));
			
			double[] values = data.attributeToDoubleArray(source);
			assert values.length == m;
			
			// reinsert at end
			data.deleteAttributeAt(source);
			data.insertAttributeAt(a, n - 1);
			assert data.attribute(n-1).equals(a);
			for (int j = 0; j < m; j++){
				if(Double.isNaN(values[j])){
					assert data.instance(j).isMissing(n-1);
				}else{
					data.instance(j).setValue(n-1, values[j]);
					assert data.instance(j).value(n-1) == values[j];
				}
			}
			// TODO: assert missingValues(values) ==
			// missingValues(data.attributeToDoubleArray(n-1));
		}
		assert attributeOrder.equals(getAttributeNames(data));

		return data;
	}

	/**
	 * Discretizes the given dataset ensuring each attribute has as many
	 * possible values as specified by the given bayesian network
	 * 
	 * @param data
	 *            the dataset to be discretized
	 * @param bn
	 *            A bayesian network whose nodes are the attributes in
	 *            {@code data}
	 * @param useEqualFrequency
	 *            Specifies whether to split the values into groups of equal
	 *            frequency ({@code true}), or into groups of equal width (
	 *            {@code false})
	 * @return a dataset where all the values have been converted to nominal,
	 *         each with as many possible values as the node of the same name in
	 *         {@code bn} had
	 * @throws IllegalArgumentException
	 *             If the bayesian network names an attribute not present in the
	 *             data
	 * @throws Exception
	 *             if {@code data} is not suitable for a
	 *             {@code weka.filters.filter}
	 * @since 0.05 2016-04-08
	 */
	public static Instances discretizeToBayes(Instances data, BayesNet bn,
			boolean useEqualFrequency) throws IllegalArgumentException,
			Exception {
		List<String> attributeNames = getAttributeNames(data);
		for (int i = 0; i < bn.getNrOfNodes(); i++) {
			int j = attributeNames.indexOf(bn.getNodeName(i));
			if (j == -1)
				throw new IllegalArgumentException("The Bayesian network "
						+ " contains attribute " + bn.getNodeName(i)
						+ " but no such attribute exists in the data set");
			Discretize d = new Discretize();
			d.setIgnoreClass(true);
			d.setAttributeIndices(Integer.toString(j + 1));
			d.setBins(bn.getCardinality(i));
			d.setUseEqualFrequency(useEqualFrequency);
			d.setInputFormat(data);
			data = Filter.useFilter(data, d);
			assert bn.getNodeName(i).equals(data.attribute(j).name());
			assert data.numDistinctValues(j) == bn.getCardinality(i);
		}
		return data;
	}

	/**
	 * Gets the names of all the attributes in a dataset
	 * 
	 * @param data
	 *            the dataset from which to get the attribute names
	 * @return A {@code List} containing the names of all the attributes such
	 *         that each position {@code i} of the list contains the name of the
	 *         attribute in {@code data.attribute(i)}
	 * @since 0.05 2016-04-08
	 */
	public static List<String> getAttributeNames(Instances data) {
		int n = data.numAttributes();
		ArrayList<String> attributeNames = new ArrayList<String>(n);
		for (int i = 0; i < n; i++)
			attributeNames.add(data.attribute(i).name());
		return attributeNames;
	}

	/**
	 * Restricts the attributes of the given data to those contained in the
	 * given {@code attribute set}
	 * 
	 * @param data
	 *            the dataset to be filtered
	 * @param attributeSet
	 *            The set of attributes to restrict the dataset to. Must be a
	 *            subset of the attributes in the dataset
	 * @return A dataset with the same number of instances, but with only the
	 *         attributes from {@code data} that were in {@code attributeSet}
	 * @throws Exception
	 *             if {@code data} is not suitable for a
	 *             {@code weka.filters.filter}
	 * @throws IllegalArgumentException
	 *             if there are attributes in {@code attributeSet} that are not
	 *             in {@code data}
	 * @since 0.05 2016-04-08
	 */
	public static Instances restrictToAttributeSet(Instances data,
			Collection<String> attributeSet) throws Exception {
		if (!getAttributeNames(data).containsAll(attributeSet))
			throw new IllegalArgumentException("The attribute set provided"
					+ " requires attributes that are not in the data set");
		int attributes = data.numAttributes();
		ArrayList<String> attributeNames = new ArrayList<String>(
				attributeSet.size());
		int checked = 0;
		Remove r = new Remove();

		for (int i = 0; i < data.numAttributes();) { // note data.numAttributes
														// will change
														// dynamically as
														// attributes are
														// removed
			String a = data.attribute(i).name();
			if (!attributeSet.contains(a)) {
				r.setAttributeIndices(Integer.toString(i + 1));
				r.setInputFormat(data);
				data = Filter.useFilter(data, r);
				assert data.attribute(a) == null; // it must have actually been
													// removed
			} else {
				attributeNames.add(a);
				i++;
			}

			checked++;
		}
		assert checked == attributes;
		assert data.numAttributes() == attributeSet.size();
		assert attributeSet.containsAll(attributeNames);
		assert attributeNames.containsAll(attributeSet);
		return data;
	}

	/**
	 * Restricts the instances of the given data to just those whose
	 * i<sup>th</sup> column matches t
	 * 
	 * @param criterion
	 *            The {@code String} to find in column {@code i}
	 * @param data
	 *            The dataset to be filtered
	 * @param i
	 *            The index of the column where it is desired to only have
	 *            {@code criterion} as the value
	 * @return a subset of {@code data} where only the rows with column
	 *         {@code i} equal to {@code criterion}
	 * @throws Exception
	 *             if {@code data} is not suitable for a
	 *             {@code weka.filters.filter}
	 * @throws IllegalArgumentException
	 *             if {@code i} is not between 0 and
	 *             {@code data.numAttributes()}
	 * @since 0.05 2016-04-08
	 */
	public static Instances filterByCriterion(String criterion, Instances data,
			int i) throws Exception {
		int originalClass = data.classIndex(); // we're going to change this so
												// store for later
		data.setClassIndex(i);
		SubsetByExpression f = new SubsetByExpression();
		f.setInputFormat(data);
		f.setExpression("CLASS is \'" + criterion + "\'"); // faster than the
															// alternative
		data = Filter.useFilter(data, f);
		data.setClassIndex(originalClass); // return to normal
		// TODO assert containsOnly(data.attribute(0), criterion)
		// Possible implementation:
		// Arrays.asList(criterion).containsAll(data.attributeToList(0))
		return data;
	}
}
