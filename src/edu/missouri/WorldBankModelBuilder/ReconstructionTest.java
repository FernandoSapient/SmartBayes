package edu.missouri.WorldBankModelBuilder;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import edu.missouri.bayesianConstructor.DomainKnowledge;
import edu.missouri.bayesianConstructor.Main;
import edu.missouri.bayesianEvaluator.BifUpdate;
import edu.missouri.bayesianEvaluator.Evaluator;
import edu.missouri.bayesianEvaluator.Trainer;
import edu.ucla.belief.BeliefNetwork;
import edu.ucla.belief.StateNotFoundException;

public class ReconstructionTest {

	/**
	 * Gets the values from this data as column vectors.
	 * 
	 * @param data
	 *            The dataset to be ordered
	 * @return A {@code Map} representing a column-majoral table, where each key
	 *         is the table's header and the list mapped to is the contents of
	 *         the column with that name.
	 * @since 0.01 2016-04-24
	 */
	public static Map<String, List<Double>> toColumns(Instances data){
		int m = data.numAttributes();
		Map<String, List<Double>> out = new HashMap<String, List<Double>>(m);

		// convert each attribute to column vectors and reinsert
		for (int i = 0; i < m; i++) {
			double[] d = data.attributeToDoubleArray(i);
			Vector<Double> col = arrayToList(d);
			
			out.put(data.attribute(i).name(), col);
		}
		return out;
	}

	/**
	 * @param d
	 * @param n
	 * @return
	 */
	public static Vector<Double> arrayToList(double[] d) {
		int n = d.length;
		Vector<Double> col = new Vector<Double>();
		for(int j=0; j<n; j++)
			col.add(new Double(d[j]));
		return col;
	}
	
	/**Adds an attribute {@code a} at {@code index} to {@code data},
	 * with the given {@code values}
	 * 
	 * @param data The data to add the new attribute to
	 * @param a The description of the new attribute
	 * @param index Where to insert the new attribute
	 * @param values A list of values to insert the attribute at
	 * 
	 * @throws IllegalArgumentException If {@code values.length} does not
	 * match the number of instances in {@code data}
	 */
	//This is a near-perfect copy of Trainer.addAttributeAt
	//TODO: Move to Trainer
	public static void addAttributeAt(Instances data, Attribute a, int index,
			List<Double> values) throws IllegalArgumentException{
		int n = values.size();
		if(data.numInstances() != n)
			throw new IllegalArgumentException("A value must be provided "
					+"for every instance ("+values.size()+" values found, "
					+data.numInstances()+" values needed)");
		data.insertAttributeAt(a, index);
		assert data.attribute(index).equals(a);
		for (int j = 0; j < n; j++) {
			if (values.get(j).isNaN()) {
				assert data.instance(j).isMissing(index);
			} else {
				data.instance(j).setValue(index, values.get(j).doubleValue());
				assert data.instance(j).value(index) == values.get(j).doubleValue();
			}
		}
	}
	/**
	 * Generates and Evaluates a bayesian network for each split. Specifically, the bayesian
	 * network's arcs and conditional probabilities are generated with the data in the
	 * map's keys, and each generated network is then tested for cross-validation
	 * with the corresponding value mapped to. The accuracy in predicting each
	 * variable is tested.
	 * <p/>
	 * The resulting accuracy is stored in a {@code Map} indexed by the name of
	 * the attribute. Accuracies are stored as {@code DoubleSummaryStatistics} to
	 * ease obtaining minimums, maximums, and averages (which are the accuracy
	 * numbers you should care about if you're doing cross-validation testing).
	 * Note that if any attribute has missing values in all of test set
	 * instances, it is not included in the result.
	 * <p/>
	 * The function adds an additional entry to the map keyed
	 * "__ProcessingTime__", for benchmarking purposes, which stores the number
	 * of seconds (in a {@code double} with nanosecond precision) it took to
	 * process each split. Note that if there is an attribute with this name,
	 * unpredictable results may be returned.
	 * 
	 * @param bn
	 *            The bayesian network to be evaluated
	 * @param splits
	 *            A {@code Map} where each key <i>k</i> is an {@code Instances}
	 *            object containing {@code ratio} of {@code data}, and {@code
	 *            get(k) returns the remaining 1 &minus; {@code ratio} of
	 *            {@code data}
	 * @param filename
	 *            The name of a file to write intermediate networks to
	 * @return A {@code Map} where each key is an attribute name (or
	 *         "__ProcessingTime__"
	 * @throws Exception
	 *             If the number of columns in data does not match the number of
	 *             nodes in {@code bn} (use
	 *             {@link Trainer#conformToNetwork(Instances, BayesNet, boolean)}
	 *             or
	 *             {@link Trainer#restrictToAttributeSet(Instances, Collection)}
	 *             for this purpose). An undocumented exception is also thrown
	 *             if the file was written correctly but could not be read
	 * @throws FileNotfoundException
	 *             If {@code filename} could not be written
	 * @throws StateNotFoundException
	 *             If the instances are not compatible with the Bayesian network
	 *             (they specify states for the nodes that are not among those
	 *             nodes' possible values)
	 * @throws UnassignedClassException
	 *             If {@code testData}'s class attribute is not set
	 * @see #randomSplit(Instances, float, int)
	 * @since 0.01 2016-04-24
	 */
	//this is a near-perfect copy of Evaluator.crossValidationAccuracies
	//TODO: make a function that makes "processing time" optional
	//TODO: recieve a file so that caller may make it a temp file
	//TODO: Move to Evaluator?
	public static Map<String, DoubleSummaryStatistics> crossValidationAccuracies(
			Map<Instances, Instances> splits,
			String filename) throws Exception, FileNotFoundException,
			StateNotFoundException, UnassignedClassException {
		int folds;
		folds = splits.size();
		Iterator<Map.Entry<Instances,Instances>> trains = splits.entrySet().iterator();
		Map<String, DoubleSummaryStatistics> results = new HashMap<String, DoubleSummaryStatistics>();
		results.put("__ProcessingTime__", new DoubleSummaryStatistics());
		while (trains.hasNext()) {
			long t = System.nanoTime();
			Map.Entry<Instances,Instances> current = trains.next();
			Instances training = current.getKey();
			constructToFile(training, filename);	
			EditableBayesNet bn = BifUpdate.loadBayesNet(filename);
			Trainer.trainToFile(bn, training, filename);
			BeliefNetwork bn1 = Evaluator.loadSamiamBayes(filename);
			Evaluator.allAttributesAccuracies(bn1, current.getValue(), results);
			
		    //String summary = wekaEvaluation(bn, current, Trainset.get(current));
			//System.out.println(summary);
			double time = (double)(System.nanoTime()-t)/TimeUnit.SECONDS.toNanos(1);
			results.get("__ProcessingTime__").accept(time);
			folds--;
			System.out.println("Processed split in "+time+" seconds; "+folds+" folds remain");
		}
		return results;
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
	 *            a filtering criterion and, at {@code args[4]}, "true" 
	 *            if frequency discrtization is desired.
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
			((CSVLoader) source.getLoader()).setMissingValue("..");
		}
		Instances data = source.getDataSet();
		
		// filter out by criterion
		if (args.length >= 4) {
			System.out.println("Filtering by " + data.attribute(0).name()
					+ " equal to " + args[3] + "...");
			data = Trainer.filterByCriterion(args[3], data, 0);
		}
		addPrevious(data, "GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]");
		addPrevious(data, "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]");
		
		//build a network
		String BN_File = args[1];
		constructToFile(data, BN_File);	//TODO referential network shouldn't need data
		
		//convert it to weka
		EditableBayesNet wekaBayes = BifUpdate.loadBayesNet(BN_File);
	
		
		
		System.out.println("Conforming data to network...");
		if (args.length >= 5)
			data = Trainer.conformToNetwork(data, wekaBayes,
					Boolean.getBoolean(args[4]));
		else
			data = Trainer.conformToNetwork(data, wekaBayes, false);
	
		float trainSize = 0.85f;
	
		// TODO move to function and receive trainSize as parameter
		assert trainSize > 0;
		assert trainSize < 1;
		
		@SuppressWarnings("deprecation")
		String date = new java.util.Date().toLocaleString();
		System.out.println("\nRESULTS\nat " + date + "\n-------");
		int folds = Math.round(1 / (1 - trainSize)) * data.attribute(0).numValues(); //assuming all attributes have the same number of values
		Map<Instances, Instances> crossVal = Evaluator.randomSplit(data, trainSize, folds);
		Map<String, DoubleSummaryStatistics> results = Evaluator.crossValidationAccuracies(
				wekaBayes, crossVal, BN_File);
		Iterator<Map.Entry<String, DoubleSummaryStatistics>> I = results.entrySet().iterator();
		while(I.hasNext()){
			Map.Entry<String, DoubleSummaryStatistics> e = I.next();
			System.out.println(e.getKey()+": min "+e.getValue().getMin()+"; max "+e.getValue().getMax()+"; average "+e.getValue().getAverage());
		}
	}

	/**
	 * @param data
	 * @param name
	 * @throws IllegalArgumentException
	 */
	public static void addPrevious(Instances data, String name)
			throws IllegalArgumentException {
		Attribute a = data.attribute(name).copy("Previous "+name);
		List<Double> PPPval = arrayToList(data.attributeToDoubleArray(Trainer.getAttributeNames(data).indexOf(name)));
		addAttributeAt(data, a, data.numAttributes()-1, ModelClusterizer.shiftBy(PPPval, 1));
	}

	/**
	 * Builds a full {@link DomainKnowledge} model following the structure
	 * proposed in the UNESCO world engineering report. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
	 * <p/>
	 * Note: this method is not a clone of {@link ModelClusterizer#buildUnescoModel(Map)}; this method assumes a
	 * the previous year variables have already been added
	 * 
	 * @param data
	 *            A {@code Map} representing a column-majoral table, where each
	 *            key is the table's header and the list mapped to is the
	 *            contents of the column with that name.
	 * @return A {@link DomainKnowledge} model with dependency tables reflecting
	 *         the relations in the given data
	 * @throws IllegalArgumentException
	 *             If the lists in {@code data} are not all the same size
	 * @since 0.04 2016-04-23
	 */
	public static DomainKnowledge buildUnescoModel(
			Map<String, List<Double>> data) throws IllegalArgumentException {
		// expected names
		String primary = "Labor force with primary education (% of total) [SL.TLF.PRIM.ZS]";
		String secondary = "Labor force with secondary education (% of total) [SL.TLF.SECO.ZS]";
		String tertiary = "Labor force with tertiary education (% of total) [SL.TLF.TERT.ZS]";
		String journal = "Scientific and technical journal articles [IP.JRN.ARTC.SC]";
		String trademark = "Trademark applications, total [IP.TMK.TOTL]";
		String government = "General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]";
		String foreignAid = "Net official development assistance and official aid received (constant 2012 US$) [DT.ODA.ALLD.KD]";
		String agriculture = "Agriculture, value added (% of GDP) [NV.AGR.TOTL.ZS]";
		String industry = "Industry, value added (% of GDP) [NV.IND.TOTL.ZS]";
		String manufacture = "Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]";
		String services = "Services, etc., value added (% of GDP) [NV.SRV.TETC.ZS]";
		String unemployed = "Unemployment, total (% of total labor force) [SL.UEM.TOTL.ZS]";
		String growth = "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]";
		String PPP = "GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]";
		String prevPPP = "Previous " + PPP;
		String prevGrowth = "Previous " + growth;

		// set categories
		List<List<Double>> education = Arrays.asList(data.get(primary),
				data.get(secondary), data.get(tertiary));
		List<List<Double>> innovation = Arrays
				.asList(data.get(journal), data.get(trademark),
						data.get(government), data.get(foreignAid));
		List<List<Double>> production = Arrays.asList(data.get(agriculture),
				data.get(industry), data.get(manufacture), data.get(services),
				data.get(unemployed));
		List<List<Double>> economic = Arrays.asList(data.get(growth),
				data.get(PPP));
		List<List<Double>> prevEcon = Arrays.asList(data.get(growth),
				data.get(PPP));

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Economic", Arrays.asList(growth, PPP));
		m.addLayer("Previous Economy",
				Arrays.asList(prevGrowth, prevPPP));
		m.addLayer("Education", Arrays.asList(primary, secondary, tertiary));
		m.addLayer("Innovation",
				Arrays.asList(journal, trademark, government, foreignAid));
		m.addLayer("Production", Arrays.asList(agriculture, industry,
				manufacture, services, unemployed));
		m.addDependency("Previous Economy", "Production",
				Main.getDependency(prevEcon, production));
		m.addDependency("Previous Economy", "Education",
				Main.getDependency(prevEcon, education));
		m.addDependency("Education", "Innovation",
				Main.getDependency(education, innovation));
		m.addDependency("Education", "Production",
				Main.getDependency(education, production));
		m.addDependency("Innovation", "Production",
				Main.getDependency(innovation, production));
		m.addDependency("Innovation", "Economic",
				Main.getDependency(innovation, economic));
		m.addDependency("Production", "Economic",
				Main.getDependency(production, economic));
		return m;
	}

	/**
	 * @param data
	 * @param BN_File
	 * @throws IllegalArgumentException
	 * @throws NullPointerException
	 * @throws FileNotFoundException
	 */
	public static void constructToFile(Instances data, String BN_File)
			throws IllegalArgumentException, NullPointerException,
			FileNotFoundException {
		DomainKnowledge m = buildUnescoModel(toColumns(data));
		String[] values = { "low", "med", "high" };	//TODO make the number of values a parameter and generate nonsense labels such as Q1,Q2,etc(training resets them anyway)
		BeliefNetwork out = Main.graphToNetwork(m.variableDependency(.03), values, m.layerMap());
		boolean result = Main.networkToFile(out, BN_File);
		assert result;
	}

}
