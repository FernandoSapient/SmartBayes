package edu.missouri.WorldBankModelBuilder;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import java.util.concurrent.TimeUnit;

import com.opencsv.CSVWriter;

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

/**
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.06 2016-04-28
 * @since {@code WorldBankModelBuilder} version 0.04 2016-04-24
 */
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
	public static Map<String, List<Double>> toColumns(Instances data) {
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
	 * @return
	 */
	public static Vector<Double> arrayToList(double[] d) {
		int n = d.length;
		Vector<Double> col = new Vector<Double>();
		for (int j = 0; j < n; j++)
			col.add(new Double(d[j]));
		return col;
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
	 */
	// This is a near-perfect copy of Trainer.addAttributeAt
	// TODO: Move to Trainer
	public static void addAttributeAt(Instances data, Attribute a, int index,
			List<Double> values) throws IllegalArgumentException {
		int n = values.size();
		if (data.numInstances() != n)
			throw new IllegalArgumentException("A value must be provided "
					+ "for every instance (" + values.size()
					+ " values found, " + data.numInstances()
					+ " values needed)");
		data.insertAttributeAt(a, index);
		assert data.attribute(index).equals(a);
		for (int j = 0; j < n; j++) {
			if (values.get(j).isNaN()) {
				assert data.instance(j).isMissing(index);
			} else {
				data.instance(j).setValue(index, values.get(j).doubleValue());
				assert data.instance(j).value(index) == values.get(j)
						.doubleValue();
			}
		}
	}

	/**
	 * Generates and Evaluates a bayesian network for each split. Specifically,
	 * the bayesian network's arcs and conditional probabilities are generated
	 * with the data in the map's keys, and each generated network is then
	 * tested for cross-validation with the corresponding value mapped to. The
	 * accuracy in predicting each variable is tested.
	 * <p/>
	 * The resulting accuracy is stored in a {@code Map} indexed by the name of
	 * the attribute. Accuracies are stored as {@code DoubleSummaryStatistics}
	 * to ease obtaining minimums, maximums, and averages (which are the
	 * accuracy numbers you should care about if you're doing cross-validation
	 * testing). Note that if any attribute has missing values in all of test
	 * set instances, it is not included in the result.
	 * <p/>
	 * The function adds an additional entry to the map keyed
	 * "__ProcessingTime__", for benchmarking purposes, which stores the number
	 * of seconds (in a {@code double} with nanosecond precision) it took to
	 * process each split. Note that if there is an attribute with this name,
	 * unpredictable results may be returned.
	 * <p/>
	 * Note: this method is not a clone of {@link Evaluator#crossValidationAccuracies(EditableBayesNet, Map, String)};
	 * This method constructs a new Bayesian network using the training data
	 * 
	 * @param splits
	 *            A {@code Map} where each key <i>k</i> is an {@code Instances}
	 *            object containing {@code ratio} of {@code data}, and {@code
	 *            get(k)} returns the remaining 1 &minus; {@code ratio} of
	 *            {@code data}
	 * @param filename
	 *            The name of a file to write intermediate networks to
	 * @param values
	 *            TODO
	 * @param useUnesco
	 *            Specifies whether to use the Unesco structure for the
	 *            domain knowledge. If {@code false}, the Smets-Woulters
	 *            structure will be used instead.
	 * 
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
	 * @throws FileNotFoundException
	 *             If {@code filename} could not be written
	 * @throws StateNotFoundException
	 *             If the instances are not compatible with the Bayesian network
	 *             (they specify states for the nodes that are not among those
	 *             nodes' possible values)
	 * @throws UnassignedClassException
	 *             If {@code testData}'s class attribute is not set
	 * @see Evaluator#randomSplit(Instances, float, int)
	 * @since 0.01 2016-04-24
	 */
	// this is a near-perfect copy of Evaluator.crossValidationAccuracies
	// TODO: make a function that makes "processing time" optional
	// TODO: recieve a file so that caller may make it a temp file
	// TODO: Move to Evaluator?
	public static Map<String, DoubleSummaryStatistics> crossValidationAccuracies(
			Map<Instances, Instances> splits, String filename, int values, boolean useUnesco)
			throws Exception, FileNotFoundException, StateNotFoundException,
			UnassignedClassException {
		int folds;
		folds = splits.size();
		Iterator<Map.Entry<Instances, Instances>> trains = splits.entrySet()
				.iterator();
		Map<String, DoubleSummaryStatistics> results = new HashMap<String, DoubleSummaryStatistics>();
		results.put("__ProcessingTime__", new DoubleSummaryStatistics());
		while (trains.hasNext()) {
			long t = System.nanoTime();
			Map.Entry<Instances, Instances> current = trains.next();
			Instances training = current.getKey();
			boolean fileCreated;
			if(useUnesco)
				fileCreated = constructUnescoToFile(training, filename,
						values);
			else
				fileCreated = constructSWToFile(training, filename,
						values);
			assert fileCreated;
			EditableBayesNet bn = BifUpdate.loadBayesNet(filename);
			Trainer.trainToFile(bn, training, filename);
			BeliefNetwork bn1 = Evaluator.loadSamiamBayes(filename);
			Evaluator.allAttributesAccuracies(bn1, current.getValue(), results);

			// String summary = wekaEvaluation(bn, current,
			// Trainset.get(current));
			// System.out.println(summary);
			double time = secondsElapsed(t);
			results.get("__ProcessingTime__").accept(time);
			folds--;
			System.out.println("\tProcessed split in " + time + " seconds; "
					+ folds + " folds remain");
		}
		return results;
	}

	/**
	 * @param sinceTime
	 * @return
	 */
	public static double secondsElapsed(long sinceTime) {
		return (double) (System.nanoTime() - sinceTime)
				/ TimeUnit.SECONDS.toNanos(1);
	}

	/**
	 * Takes a (trained?) Bayesian network (provided in an XML BIF file) and
	 * evaluates it using the given data. <!--The following is copied from
	 * Trainer.main--> The data file must be on one of Weka's accepted file
	 * formats (ARFF, C4.5, CSV, JSON, LibSVM, MatLab, DAT, BSI, or XRFF, as of
	 * Weka version 3.7.12). The program might fail if the Bayesian network is
	 * not in Weka format (use {@link BifUpdate} for this purpose). A filtering
	 * criterion can be specified, against which only the elements of the first
	 * attribute which are equal to it will be selected.
	 * <p/>
	 * The method discretizes each attribute to match the number of possible
	 * values in its corresponding node (see
	 * {@link Trainer#discretizeToBayes(Instances, BayesNet, boolean)}) unless the
	 * attribute is already discrete.
	 * <p/>
	 * If the file is CSV file, the data is assumed to have originated from the
	 * World Bank, and the missing value placeholder is set to "..", but this
	 * may change in a future version.
	 * 
	 * @param args
	 *            An array containing, at {@code args[0]}, the path of the data
	 *            file to train the Bayesian network with; at {@code args[1]}
	 *            the path of the file to write the output tables to, at
	 *            {@code args[2]}, a folder in which to save intermediate
	 *            bayesian networks; the path of the file containing the
	 *            Bayesian network to train; at {@code args[2]} at
	 *            {@code args[3]} whether to use the unesco model ({@code true})
	 *            or the Smets-Wouters model ({@code false})the path of the
	 *            output file in which to store the trained network; and
	 *            optionally, at {@code args[4]} the index with the values to
	 *            group the results by (zero by default); at {@code args[5]} the
	 *            number of bins to use in discretization, and at
	 *            {@code args[6]}, "true" if frequency discrtization is desired.
	 * @throws Exception
	 *             If any of the files could not be read
	 * @since {@code bayesianEvaluator} 0.01 2016-04-10
	 */
	// TODO improve parameter handling. See JCLAP for potential solution
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err
					.println("Usage: java RecontructionTest <input data file> <output directory> <working directory> <Use Unesco Model> <group-by column> [number of discrete values] [use equal frequency]");
			return;
		}
		String filename = args[0];
		String outFile = args[1];
		String dir = args[2];
		boolean useUnesco = Boolean.getBoolean(args[3]);
		int groupByIndex = Integer.parseInt(args[4]);
		int values;
		if (args.length > 5)
			values = Integer.parseInt(args[5]);
		else
			values = 3;
		boolean useEqualFrequency;
		if (args.length > 6)
			useEqualFrequency = Boolean.getBoolean(args[6]);
		else
			useEqualFrequency = false;
		float trainSize = 0.85f;
		int folds = 20;

		Set<String> countries = ModelClusterizer.getCountries(filename,
				groupByIndex);
		int totalCountries = countries.size();
		Iterator<String> C = countries.iterator();

		DataSource source = new DataSource(filename);
		if (source.getLoader() instanceof CSVLoader) {
			((CSVLoader) source.getLoader()).setMissingValue("..");
		}
		Instances data = source.getDataSet();
		CSVWriter minCSV = new CSVWriter(new FileWriter(outFile + "/min.csv"),
				',');
		CSVWriter maxCSV = new CSVWriter(new FileWriter(outFile + "/max.csv"),
				',');
		CSVWriter avgCSV = new CSVWriter(new FileWriter(outFile + "/avg.csv"),
				',');
		List<String> titles;
		if(useUnesco){
			minCSV.writeNext(getUnescoTitles());
			maxCSV.writeNext(getUnescoTitles());
			avgCSV.writeNext(getUnescoTitles());
			titles = Arrays.asList(getUnescoTitles());
		}else{
			minCSV.writeNext(getSWTitles());
			maxCSV.writeNext(getSWTitles());
			avgCSV.writeNext(getSWTitles());
			titles = Arrays.asList(getSWTitles());
		}
		minCSV.close();
		maxCSV.close();
		avgCSV.close();

		while (C.hasNext()) {
			long t = System.nanoTime();
			String country = C.next().trim();
			// filter out by criterion
			Instances countryData = Trainer.filterByCriterion(country, data,
					groupByIndex);
			if(useUnesco){
				Trainer.addShifted(countryData,
						"Previous ", "GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]", 1);
				Trainer.addShifted(countryData,
						"Previous ", "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]", 1);
			}else{
				Trainer.addShifted(countryData,
						"Previous ", "Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]", 1);
				Trainer.addShifted(countryData,
						"Previous ", "Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]", 1);
				Trainer.addShifted(countryData,
						"Previous ", "Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]", 1);
				Trainer.addShifted(countryData,
						"Previous ", "Compensation of employees (current LCU) [GC.XPN.COMP.CN]", 1);
				Trainer.addShifted(countryData,
						"Previous ", "Gross capital formation (current LCU) [NE.GDI.TOTL.CN]", 1);
			}
			
			// build a referential network
			String BN_File = dir + "/" + country + ".xml";
			boolean fileCreated;
			if(useUnesco)
				fileCreated = constructEmptyUnescoToFile(countryData,
						BN_File, values);
			else
				fileCreated = constructEmptySWToFile(countryData,
						BN_File, values);
			if (fileCreated)
				System.out.println("File \"" + BN_File
						+ "\" created successfully");
			else
				System.err.println("Could not write file \"" + BN_File + "\".");

			// convert it to weka
			EditableBayesNet wekaBayes = BifUpdate.loadBayesNet(BN_File);

			// TODO: discretize AFTER making the training nets
			try {
				countryData = Trainer.conformToNetwork(countryData, wekaBayes,
						useEqualFrequency);

				// TODO move to function and receive trainSize as parameter
				@SuppressWarnings("deprecation")
				String date = new java.util.Date().toLocaleString();
				System.out.println("\nRESULTS\nat " + date + "\n-------");
				Map<Instances, Instances> crossVal = Evaluator.randomSplit(
						countryData, trainSize, folds);
				Map<String, DoubleSummaryStatistics> results = crossValidationAccuracies(
						crossVal, BN_File, values, useUnesco);
				double time = secondsElapsed(t);
				Iterator<Map.Entry<String, DoubleSummaryStatistics>> I = results
						.entrySet().iterator();
				String[] mins = new String[titles.size()];
				String[] maxs = new String[titles.size()];
				String[] avgs = new String[titles.size()];
				minCSV = new CSVWriter(new FileWriter(outFile + "/min.csv",
						true), ',', CSVWriter.NO_QUOTE_CHARACTER);
				maxCSV = new CSVWriter(new FileWriter(outFile + "/max.csv",
						true), ',', CSVWriter.NO_QUOTE_CHARACTER);
				avgCSV = new CSVWriter(new FileWriter(outFile + "/avg.csv",
						true), ',', CSVWriter.NO_QUOTE_CHARACTER);
				mins[0] = country;
				maxs[0] = country;
				avgs[0] = country;
				System.out.println(country + ":");
				while (I.hasNext()) {
					Map.Entry<String, DoubleSummaryStatistics> e = I.next();
					int i = titles.indexOf(e.getKey());
					mins[i] = String.valueOf(e.getValue().getMin());
					maxs[i] = String.valueOf(e.getValue().getMax());
					avgs[i] = String.valueOf(e.getValue().getAverage());
					System.out.println("\t" + e.getKey() + ": min "
							+ e.getValue().getMin() + "; max "
							+ e.getValue().getMax() + "; average "
							+ e.getValue().getAverage());
				}
				System.out.println("Processed " + country + " in " + time
						+ " seconds");
				minCSV.writeNext(mins);
				maxCSV.writeNext(maxs);
				avgCSV.writeNext(avgs);
				minCSV.close();
				maxCSV.close();
				avgCSV.close();
			} catch (ArithmeticException e) {
				System.out.println("Skipping " + country
						+ ": Insufficient data: " + e);
			}
			// results.put("__TotalTime__", new DoubleSummaryStatistics(time) );
			totalCountries--;
			System.out.println(totalCountries + " countries remain");
		}
	}

	/**
	 * Makes a copy of the attribute {@code name}, preceeded by "Previous ",
	 * where all valeus are shifted to the next instance. This is useful
	 * for time series data when indicating what the value was at the
	 * previous point in time might help.
	 * 
	 * @param data
	 *            {@code Instances} set where attribute {@code name} lives, and
	 *            to which the new attribute will be added
	 * @param name
	 *            The name of the attribute to be copied and shifted
	 * @deprecated since 0.06 2016-04-28 (all methods in main programs are
	 *             subject to be moved!) Use
	 *             <a href="../Trainer#addShifted(Instances,String,String, int)">{@code Trainer.addShifted}</a>{@code (data, "Previous ", name, 1}
	 *             instead
	 */
	public static void addPrevious(Instances data, String name){
				Trainer.addShifted(data, "Previous ", name, 1);
			}

	/**
	 * Builds a full {@link DomainKnowledge} model following the structure
	 * proposed in the UNESCO world engineering report. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
	 * <p/>
	 * Note: this method is not a clone of
	 * {@link ModelClusterizer#buildUnescoModel(Map)}; this method assumes a the
	 * previous year variables have already been added
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

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Economic", Arrays.asList(growth, PPP));
		m.addLayer("Previous Economy", Arrays.asList(prevGrowth, prevPPP));
		m.addLayer("Education", Arrays.asList(primary, secondary, tertiary));
		m.addLayer("Innovation",
				Arrays.asList(journal, trademark, government, foreignAid));
		m.addLayer("Production", Arrays.asList(agriculture, industry,
				manufacture, services, unemployed));
		
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
	 * @param filename
	 * @param values
	 *            TODO
	 * @throws IllegalArgumentException
	 * @throws NullPointerException
	 * @throws FileNotFoundException
	 */
	public static boolean constructUnescoToFile(Instances data,
			String filename, int values) throws IllegalArgumentException,
			NullPointerException, FileNotFoundException {
		DomainKnowledge m = buildUnescoModel(toColumns(data));
		BeliefNetwork out = Main.graphToNetwork(m.variableDependency(.03),
				Main.genValues(values), m.layerMap());
		return Main.networkToFile(out, filename);
	}
	
	/**
	 * 
	 * @param data
	 * @param filename
	 * @param values
	 *            TODO
	 * @return
	 * @throws IllegalArgumentException
	 * @throws NullPointerException
	 * @throws FileNotFoundException
	 * @since 0.05 2016-04-27 
	 */
	public static boolean constructSWToFile(Instances data,
			String filename, int values) throws IllegalArgumentException,
			NullPointerException, FileNotFoundException {
		DomainKnowledge m = buildSWModel(toColumns(data));
		BeliefNetwork out = Main.graphToNetwork(m.variableDependency(.03),
				Main.genValues(values), m.layerMap());
		return Main.networkToFile(out, filename);
	}

	/**
	 * @param data
	 * @param filename
	 * @param values
	 *            TODO
	 * @throws IllegalArgumentException
	 * @throws NullPointerException
	 * @throws FileNotFoundException
	 * @since 0.05 2016-04-27
	 */
	public static boolean constructEmptyUnescoToFile(Instances data,
			String filename, int values) throws IllegalArgumentException,
			NullPointerException, FileNotFoundException {
		DomainKnowledge m = buildEmptyUnescoModel(toColumns(data));
		BeliefNetwork out = Main.graphToNetwork(m.variableDependency(.03),
				Main.genValues(values), m.layerMap());
		return Main.networkToFile(out, filename);
	}
	
	/**
	 * 
	 * @param data
	 * @param filename
	 * @param values
	 * @return
	 * @throws IllegalArgumentException
	 * @throws NullPointerException
	 * @throws FileNotFoundException
	 * @since 0.05 2016-04-27
	 */
	public static boolean constructEmptySWToFile(Instances data,
			String filename, int values) throws IllegalArgumentException,
			NullPointerException, FileNotFoundException {
		DomainKnowledge m = buildEmptySWModel(toColumns(data));
		BeliefNetwork out = Main.graphToNetwork(m.variableDependency(.03),
				Main.genValues(values), m.layerMap());
		return Main.networkToFile(out, filename);
	}

	/**
	 * Builds a full {@link DomainKnowledge} model following the Smets-Woulters
	 * standard economic model. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
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
	public static DomainKnowledge buildSWModel(
			Map<String, List<Double>> data) throws IllegalArgumentException {
		// expected names
		String consump = "Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]";
		String worker = "Wage and salaried workers, total (% of total employed) [SL.EMP.WORK.ZS]";
		String interest = "Lending interest rate (%) [FR.INR.LEND]";
		String invest = "Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]";
		String capital = "Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]";
		String form = "Gross capital formation (current LCU) [NE.GDI.TOTL.CN]";
		String GDP = "GDP (constant LCU) [NY.GDP.MKTP.KN]";
		String exog = "Exogenous spending";
		String wages = "Compensation of employees (current LCU) [GC.XPN.COMP.CN]";
		String inflation = "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]";
		String ratio = "Capital-labour ratio";
		String prevConsump = "Previous Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]";
		String prevInvest = "Previous Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]";
		String prevCapital = "Previous Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]";
		String prevWages = "Previous Compensation of employees (current LCU) [GC.XPN.COMP.CN]";
		String prevForm = "Previous Gross capital formation (current LCU) [NE.GDI.TOTL.CN]";

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Resource", Arrays.asList(wages, interest, form, ratio));
		m.addLayer("Estimation", Arrays.asList(consump, invest, worker, capital, exog));
		m.addLayer("Economy", Arrays.asList(GDP));
		m.addLayer("PrevEstimation", Arrays.asList(prevConsump, prevInvest, prevCapital));
		m.addLayer("PrevResource", Arrays.asList(inflation, prevWages, prevForm));
		// set categories
		List<List<Double>> resource = Arrays.asList(data.get(wages),
				data.get(interest), data.get(form), data.get(ratio));
		List<List<Double>> estimation = Arrays
				.asList(data.get(consump), data.get(invest),
						data.get(worker), data.get(capital), data.get(exog));
		List<List<Double>> economy = Arrays.asList(data.get(GDP));
		List<List<Double>> prevEstimation = Arrays.asList(data.get(prevConsump),
				data.get(prevInvest), data.get(prevCapital));
		List<List<Double>> prevResource = Arrays.asList(data.get(inflation),
				data.get(prevWages), data.get(prevForm));

		// Hardwire 3-layer structure
		m.addDependency("PrevResource", "Resource",
				Main.getDependency(prevResource, resource));
		m.addDependency("Resource", "Estimation",
				Main.getDependency(resource, estimation));
		m.addDependency("PrevEstimation", "Estimation",
				Main.getDependency(prevEstimation, estimation));
		m.addDependency("Estimation", "Economy",
				Main.getDependency(estimation, economy));
		return m;
	}
	
	/**
	 * Builds a {@link DomainKnowledge} model following the structure
	 * proposed in the UNESCO world engineering report without any dependencies.
	 * This method is included as sample code and will be deprecated as soon as
	 * file-reading support is added
	 * <p/>
	 * Note: this method is not a clone of
	 * {@link ModelClusterizer#buildUnescoModel(Map)}; this method assumes a the
	 * previous year variables have already been added
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
	public static DomainKnowledge buildEmptyUnescoModel(
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

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Economic", Arrays.asList(growth, PPP));
		m.addLayer("Previous Economy", Arrays.asList(prevGrowth, prevPPP));
		m.addLayer("Education", Arrays.asList(primary, secondary, tertiary));
		m.addLayer("Innovation",
				Arrays.asList(journal, trademark, government, foreignAid));
		m.addLayer("Production", Arrays.asList(agriculture, industry,
				manufacture, services, unemployed));
		return m;
	}

	public static String[] getUnescoTitles() {
		String[] out = {
				"Country",
				"Labor force with primary education (% of total) [SL.TLF.PRIM.ZS]",
				"Labor force with secondary education (% of total) [SL.TLF.SECO.ZS]",
				"Labor force with tertiary education (% of total) [SL.TLF.TERT.ZS]",
				"Scientific and technical journal articles [IP.JRN.ARTC.SC]",
				"Trademark applications, total [IP.TMK.TOTL]",
				"General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]",
				"Net official development assistance and official aid received (constant 2012 US$) [DT.ODA.ALLD.KD]",
				"Agriculture, value added (% of GDP) [NV.AGR.TOTL.ZS]",
				"Industry, value added (% of GDP) [NV.IND.TOTL.ZS]",
				"Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]",
				"Services, etc., value added (% of GDP) [NV.SRV.TETC.ZS]",
				"Unemployment, total (% of total labor force) [SL.UEM.TOTL.ZS]",
				"GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]",
				"GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]",
				"Previous GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]",
				"Previous GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]",
				"__ProcessingTime__" };
		return out;
	}
	
	/**
	 * 
	 * @return
	 * @since 0.05 2016-04-27
	 */
	public static String[] getSWTitles() {
		String[] out = {
				"Country",
				"Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]",
				"Wage and salaried workers, total (% of total employed) [SL.EMP.WORK.ZS]",
				"Lending interest rate (%) [FR.INR.LEND]",
				"Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]",
				"Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]",
				"Gross capital formation (current LCU) [NE.GDI.TOTL.CN]",
				"GDP (constant LCU) [NY.GDP.MKTP.KN]",
				"Exogenous spending",
				"Compensation of employees (current LCU) [GC.XPN.COMP.CN]",
				"Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]",
				"Capital-labour ratio",
				"Previous Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]",
				"Previous Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]",
				"Previous Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]",
				"Previous Compensation of employees (current LCU) [GC.XPN.COMP.CN]",
				"Previous Gross capital formation (current LCU) [NE.GDI.TOTL.CN]",
				"__ProcessingTime__" };
	return out;
	}

	/**
	 * Builds a full {@link DomainKnowledge} model following the Smets-Woulters
	 * standard economic model without any dependencies. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
	 * 
	 * @param data
	 *            A {@code Map} representing a column-majoral table, where each
	 *            key is the table's header and the list mapped to is the
	 *            contents of the column with that name.
	 * @return A {@link DomainKnowledge} model with dependency tables reflecting
	 *         the relations in the given data
	 * @throws IllegalArgumentException
	 *             If the lists in {@code data} are not all the same size
	 * @since 0.05 2016-04-27
	 */
	public static DomainKnowledge buildEmptySWModel(
			Map<String, List<Double>> data) throws IllegalArgumentException {
		// expected names
		String consump = "Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]";
		String worker = "Wage and salaried workers, total (% of total employed) [SL.EMP.WORK.ZS]";
		String interest = "Lending interest rate (%) [FR.INR.LEND]";
		String invest = "Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]";
		String capital = "Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]";
		String form = "Gross capital formation (current LCU) [NE.GDI.TOTL.CN]";
		String GDP = "GDP (constant LCU) [NY.GDP.MKTP.KN]";
		String exog = "Exogenous spending";
		String wages = "Compensation of employees (current LCU) [GC.XPN.COMP.CN]";
		String inflation = "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]";
		String ratio = "Capital-labour ratio";
		String prevConsump = "Previous Final consumption expenditure (constant LCU) [NE.CON.TOTL.KN]";
		String prevInvest = "Previous Portfolio Investment, net (BoP, current US$) [BN.KLT.PTXL.CD]";
		String prevCapital = "Previous Net capital account (BoP, current US$) [BN.TRF.KOGT.CD]";
		String prevWages = "Previous Compensation of employees (current LCU) [GC.XPN.COMP.CN]";
		String prevForm = "Previous Gross capital formation (current LCU) [NE.GDI.TOTL.CN]";

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Resource", Arrays.asList(wages, interest, form, ratio));
		m.addLayer("Estimation", Arrays.asList(consump, invest, worker, capital, exog));
		m.addLayer("Economy", Arrays.asList(GDP));
		m.addLayer("PrevEstimation", Arrays.asList(prevConsump, prevInvest, prevCapital));
		m.addLayer("PrevResource", Arrays.asList(inflation, prevWages, prevForm));
		return m;
	}

}
