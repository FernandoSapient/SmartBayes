package edu.missouri.bayesianEvaluator;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import edu.missouri.WorldBankModelBuilder.ReconstructionTest;
import edu.missouri.bayesianConstructor.Main;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.EditableBayesNet;
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
 * <p/>
 * The {@link #main(String[])} program can be called on an existing bayesian
 * network and data file by using
 * {@code java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion] [UseFrequencyDiscretization]}
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.15 2016-04-28
 * @since {@code bayesianEvaluator} version 0.02 2016-04-02
 */
// TODO: create non-static versions of all methods
// (store the bayesian network you're training as an attribute)
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
		// see also attributeStats
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
	 * Adds an attribute {@code a} at {@code index} to {@code data}, with the
	 * given {@code values}
	 * 
	 * @param data
	 *            The data to add the new attribute to
	 * @param a
	 *            The description of the new attribute
	 * @param index
	 *            Where to insert the new attribute
	 * @param values
	 *            A list of values to insert the attribute at
	 * 
	 * @throws IllegalArgumentException
	 *             If {@code values.length} does not match the number of
	 *             instances in {@code data}
	 * @since 0.11 2016-04-24
	 */
	public static void addAttributeAt(Instances data, Attribute a, int index,
			double[] values) throws IllegalArgumentException {
		if (data.numInstances() != values.length)
			throw new IllegalArgumentException("A value must be provided "
					+ "for every instance (" + values.length
					+ " values found, " + data.numInstances()
					+ " values needed)");
		data.insertAttributeAt(a, index);
		assert data.attribute(index).equals(a);
		for (int j = 0; j < values.length; j++) {
			if (Double.isNaN(values[j])) {
				assert data.instance(j).isMissing(index);
			} else {
				data.instance(j).setValue(index, values[j]);
				assert data.instance(j).value(index) == values[j];
			}
		}

		assert Arrays.equals(values, data.attributeToDoubleArray(index));
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
					"The number of attributes in the ordering does not match"
							+ " the number of attributes in the data");
		List<String> attributeNames = getAttributeNames(data);

		// test "equality" by double contention
		if (!attributeOrder.containsAll(attributeNames))
			throw new IllegalArgumentException("There are attributes in the"
					+ " dataset not present in the proposed ordering");
		if (!attributeNames.containsAll(attributeOrder))
			throw new IllegalArgumentException("There are attributes in the"
					+ " proposed ordering not present in the dataset");

		// convert each attribute to column vectors and reinsert
		for (int i = 0; i < n; i++) {
			attributeNames = getAttributeNames(data); // the order of the
														// attribute names will
														// change dynamically
			int source = attributeNames.indexOf(attributeOrder.get(i));
			assert source != -1; // if it is, double contention test was wrong

			Attribute a = data.attribute(source);
			assert a.name().equals(attributeOrder.get(i));

			double[] values = data.attributeToDoubleArray(source);
			assert data.numInstances() == values.length;

			// reinsert at end
			data.deleteAttributeAt(source);
			addAttributeAt(data, a, n - 1, values);
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
	 * @throws ArithmeticException
	 *             If any one of the attributes does not have enough data to be
	 *             discretized into the number of bins in the corresponding node
	 *             in {@code bn} (e.g.: attempting to put 2 data points into 3
	 *             bins)
	 * @throws AssertionError
	 *             If any one of the attributes of {@code data} is not
	 *             discretizable by
	 *             {@code weka.filters.unsupervised.attribute.Discretize} (i.e.
	 *             is already discrete or is a text attribute)
	 * @throws Exception
	 *             if {@code data} is not suitable for a
	 *             {@code weka.filters.filter}
	 * @since 0.05 2016-04-08
	 */
	public static Instances discretizeToBayes(Instances data, BayesNet bn,
			boolean useEqualFrequency) throws IllegalArgumentException,
			ArithmeticException, AssertionError, Exception {
		List<String> attributeNames = getAttributeNames(data);
		for (int i = 0; i < bn.getNrOfNodes(); i++) {
			int j = attributeNames.indexOf(bn.getNodeName(i));
			if (j == -1)
				throw new IllegalArgumentException("The Bayesian network "
						+ " contains attribute " + bn.getNodeName(i)
						+ " but no such attribute exists in the data set");
			assert (data.attribute(j).isNumeric());
			if (data.attributeStats(j).missingCount == data.numInstances()
					|| data.numDistinctValues(j) <= bn.getCardinality(i)) {
				Attribute a = new Attribute(data.attribute(j).name(),
						Arrays.asList(Main.genValues(bn.getCardinality(i))));
				data.deleteAttributeAt(j);
				data.insertAttributeAt(a, j);
				;
			}/*
			 * else if(data.numDistinctValues(j) <= bn.getCardinality(i)){
			 * //TODO: add a handling case here (for 2 values, 3 bins,
			 * //the ranges should be (-inf val1](val1 val2)[val2 +inf]) 
			 * throw new ArithmeticException
			 * (bn.getNodeName(i) + " requires discretizing into "
			 * + bn.getCardinality
			 * (i) + " values, but only " + data.numDistinctValues(j)
			 * + " distinct values were found in the data");
			 * 
			 * }
			 */else {
				Discretize d = new Discretize();
				d.setIgnoreClass(true);
				d.setAttributeIndices(Integer.toString(j + 1));
				d.setBins(bn.getCardinality(i));
				d.setUseEqualFrequency(useEqualFrequency);
				d.setInputFormat(data);
				data = Filter.useFilter(data, d);
			}
			assert bn.getNodeName(i).equals(data.attribute(j).name());
			assert data.attribute(j).isNominal();
			assert data.numDistinctValues(j) == bn.getCardinality(i);
		}
		return data;
	}

	/**
	 * Conforms the {@code data} to be succesfully used by
	 * {@code weka.classifiers.bayes.BayesNet.estimateCPTs()} to train the given
	 * Bayesian network
	 * 
	 * @param data
	 *            The data to adapt to the bayesian network
	 * @param bn
	 *            The Bayesian network the data will be used with
	 * @param useEqualFrequency
	 *            Specifies whether to split the values into groups of equal
	 *            frequency (see
	 *            {@link #discretizeToBayes(Instances, BayesNet, boolean)})
	 * @return a dataset where the attributes have been restricted to, and
	 *         ordered to match, the given bayesian network, and the values of
	 *         those attributes have been discretized to match the number of
	 *         values specified in the bayesian network
	 * @throws IllegalArgumentException
	 *             If the bayesian network names an attribute not present in the
	 *             data
	 * @throws AssertionError
	 *             If any one of the attributes of {@code data} is not
	 *             discretizable by
	 *             {@code weka.filters.unsupervised.attribute.Discretize} (i.e.
	 *             is already discrete or is a text attribute)
	 * @throws Exception
	 *             if {@code data} is not suitable for a
	 *             {@code weka.filters.filter}
	 * @since 0.08 2016-04-10
	 */
	public static Instances conformToNetwork(Instances data, BayesNet bn,
			boolean useEqualFrequency) throws Exception, AssertionError,
			ArithmeticException, IllegalArgumentException {
		List<String> nodes = getNodeNames(bn);

		// remove attributes not in the bayesian network
		data = restrictToAttributeSet(data, nodes);

		// reorder to match (required by estimateCPTs)
		data = reorderAttributes(data, nodes);

		// discretize data according to BN value labels
		data = discretizeToBayes(data, bn, useEqualFrequency);
		return data;
	}

	/**
	 * Trains the given network with the given data and stores the result with
	 * the given filename. Note that {@code bn} is modified by this function.
	 * Also note that if the columns in {@code data} are not already in the same
	 * order as the nodes in {@code bn}, unpredictable behavior may
	 * result&mdash;use {@link #conformToNetwork(Instances, BayesNet, boolean)}
	 * or {@link #reorderAttributes(Instances, List)} for this purpose.
	 * <p/>
	 * To train the Bayesian network without generating a file, call
	 * 
	 * @param bn
	 *            The Bayesian network to be trained
	 * @param data
	 *            The data with which to train the Bayesian network
	 * @param filename
	 *            The name of the file to save the Bayesian network in
	 * @throws Exception
	 *             If the number of columns in {@code data} does not match the
	 *             number of nodes in {@code bn} (use
	 *             {@link #conformToNetwork(Instances, BayesNet, boolean)} or
	 *             {@code #restrictToAttributeSet(Instances, Collection)} for
	 *             this purpose)
	 * @throws FileNotFoundException
	 *             if {@code filename} could not be created
	 * @since 0.10 2016-04-20
	 */
	// TODO: create overload method that receives a file
	public static void trainToFile(EditableBayesNet bn, Instances data,
			String filename) throws Exception, FileNotFoundException {
		bn.setData(data);
		bn.estimateCPTs();

		PrintWriter f = new PrintWriter(filename);
		f.write(bn.toXMLBIF03());
		f.close();
	}

	/**
	 * Creates a copy of the list where all element indices are shifted ahead by
	 * the amount indicated. Specifically, the value at each position
	 * {@code list.get(j)} will be found in position {@code j+i} in the returned
	 * list.
	 * <p/>
	 * The size of the list is preserved: Elements near the end are dropped off,
	 * and {@code NaN} is used as a filler element
	 * <p/>
	 * Note that, for now, {@code i} should be positive; support for negative
	 * shifts may be added at a later date
	 * 
	 * @param list
	 *            The list to be shifted
	 * @param i
	 *            The number of positions to shift the contents of {@code list}
	 *            by
	 * @return A {@code List} where the first {@code i} positions are
	 *         {@code null} and, excepting the last {@code i} positions of
	 *         {@code list}, all the elements from {@code list} are in it
	 * @since 0.15 2016-04-28
	 */
	public static List<Double> shiftBy(List<Double> list, int i) {
		List<Double> out = new Vector<Double>(list);
		out.remove(list.size() - 1);
		if (i > 0) {
			for (; i > 0; i--)
				out.add(0, new Double(Double.NaN));
			return out;
		} else {
			throw new UnsupportedOperationException(
					"No support yet for negative i");
			// TODO (but is it worth it?)
		}
	}

	/**
	 * Makes a copy of the attribute {@code name} shifting all its values by
	 * {@code amount}
	 * 
	 * @param data
	 *            {@code Instances} set where attribute {@code name} lives, and
	 *            to which the new attribute will be added
	 * @param prefix
	 *            Prefix to add to the name of the attribute to distinguish it
	 *            from {@code name}
	 * @param name
	 *            The name of the attribute to be copied and shifted
	 * @param amount
	 *            number of instance positions by which to shift the values by.
	 * @throws ArrayIndexOutOfBoundsException
	 *             if {@code name} does not exist in {@code data}
	 * @see #shiftBy(List, int)
	 */
	public static void addShifted(Instances data, String prefix, String name,
			int amount) throws ArrayIndexOutOfBoundsException {
		Attribute a = data.attribute(name).copy(prefix + name);
		List<String> atts = getAttributeNames(data);
		int index = atts.indexOf(name);
		List<Double> values = ReconstructionTest.arrayToList(data
				.attributeToDoubleArray(index));
		ReconstructionTest.addAttributeAt(data, a, data.numAttributes() - 1,
				shiftBy(values, amount));
	}

	/**
	 * Finds all the nodes in {@code bn} that start with the given
	 * {@code prefix} and fills them out with the data from an attribute with a
	 * name matching the removal of said prefix
	 * 
	 * @param data
	 *            {@code Instances} where the attribute with the unshifted data
	 *            lives and to which the shifted data must be added
	 * @param bn
	 *            Bayesian network containing the names of the nodes {@cod data}
	 *            must have
	 * @param prefix
	 *            A prefix with which shifted attirbute names must start, to
	 *            distinguish them from unshifted attributes
	 * @param amount
	 *            number of instance positions by which to shift the values by.
	 * @throws ArrayIndexOutOfBoundsException
	 *             if an attribute with the shift {@cod prefix} exists in the
	 *             Bayesian network, but the corresponding unshifted data does
	 *             not exist in {@code data}
	 */
	public static void detectAndAddAllShifted(Instances data,
			EditableBayesNet bn, String prefix, int amount)
			throws ArrayIndexOutOfBoundsException {
		List<String> names = getNodeNames(bn);
		for (int i = names.size() - 1; i >= 0; i--) {
			if (names.get(i).startsWith(prefix)) {
				String originalName = names.get(i).substring(prefix.length());
				// If we ever use column vectors again:
				// data.put(names.get(i), Trainer.shiftBy(data.get(i),
				// amount));
				addShifted(data, prefix, originalName, amount);
			}
		}
	}

	/**
	 * Trains a Bayesian network (provided in an XML BIF file) using the given
	 * data. The data file must be on one of Weka's accepted file formats (ARFF,
	 * C4.5, CSV, JSON, LibSVM, MatLab, DAT, BSI, or XRFF, as of Weka version
	 * 3.7.12). The program might fail if the Bayesian network is not in Weka
	 * format (use {@link BifUpdate} for this purpose). A filtering criterion
	 * can be specified, against which only the elements of the first attribute
	 * which are equal to it will be selected.
	 * <p/>
	 * The method discretizes each attribute to match the number of possible
	 * values in its corresponding node (see
	 * {@link #discretizeToBayes(Instances, BayesNet, boolean)}) unless the
	 * attribute is already discrete.
	 * <p/>
	 * If the file is CSV file, the data is assumed to have originated from the
	 * World Bank, and the missing value placeholder is set to "..", but this
	 * may change in a future version.
	 * 
	 * @param args
	 *            An array containing, at {@code args[0]}, the pat of the data
	 *            file to train the Bayesian network with; at {@code args[1]}
	 *            the path of the file containing the Bayesian network to train;
	 *            at at {@code args[2]} the path of the output file in which to
	 *            store the trained network; and optionally, at {@code args[3]}
	 *            a filtering criterion, at {@code args[4]}, "true" if frequency
	 *            discretization is desired, at {@code args[5]} a prefix to
	 *            identify attributes that require shifting to create, and at
	 *            {@code args[6]} the amount by which this shift should be.
	 * @throws Exception
	 *             If any of the files could not be read
	 * @since 0.01 2016-04-02
	 */
	// TODO figure out a way to parse any regular expression as filter criterion
	// TODO improve parameter handling. See JCLAP for potential solution
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java Trainer <input data file> <input XMLBIF file> <output XMLBIF file> [Filter criterion] [UseFrequencyDiscretization] [Shift Prefix] [Shift ammount]");
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
		if (args.length > 3) {
			System.out.println("Filtering by " + data.attribute(0).name()
					+ " equal to " + args[3] + "...");
			data = filterByCriterion(args[3], data, 0);
		}

		// Add shifted attributes
		if (args.length > 5) {
			String prefix = args[5];
			int amount;
			if (args.length > 6)
				amount = Integer.parseInt(args[6]);
			else
				amount = 1;
			detectAndAddAllShifted(data, bn, prefix, amount);
			// TODO: create a conformToNework overload method that does this
			// when it recieves the prefix and adjust WorldBankModelBuilders
			// to use it accordingly
		}

		System.out.println("Conforming data to network...");
		if (args.length > 4)
			data = conformToNetwork(data, bn, Boolean.parseBoolean(args[4]));
		else
			data = conformToNetwork(data, bn, false);

		data.setClassIndex(data.numAttributes() - 1);
		System.out.println("Setting " + data.classAttribute().name()
				+ " as class...");

		System.out.println("Training network...");
		trainToFile(bn, data, args[2]);
		System.out.println(args[2] + " created successfully");
	}
}
