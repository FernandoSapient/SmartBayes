/**
 * 
 */
package edu.missouri.bayesianEvaluator;

import java.io.File;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import edu.ucla.belief.BeliefNetwork;
import edu.ucla.belief.EliminationHeuristic;
import edu.ucla.belief.FiniteVariable;
import edu.ucla.belief.InferenceEngine;
import edu.ucla.belief.StateNotFoundException;
import edu.ucla.belief.Table;
import edu.ucla.belief.io.NodeLinearTask;
import edu.ucla.belief.io.PropertySuperintendent;
import edu.ucla.belief.io.xmlbif.SkimmerEstimator;
import edu.ucla.belief.io.xmlbif.XmlbifParser;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Allows evaluating a bayesian network with a dataset
 *
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.08 2016-04-23
 * @since {@code bayesianEvaluator} version 0.10 2016-04-19
 */
// TODO: create non-static versions of all methods
// (store the bayesian network you're training as an attribute)
// TODO: have a class of weka-only evaluation and a class of samiam-only evaluation
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
	 * @since 0.01 2016-04-19
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
	
	//TODO: loadEvidenceFrom ...anything else
	/**
	 * Sets the specified Bayesian Network to the state specified
	 * in {@code evidence}. Note that the class value is assumed to be the query
	 * target, and is therefore omitted. 
	 * <p/>
	 * The Bayesian network provided is modified by this method. It's current
	 * state is completely removed and replaced with the state given by
	 * the evidence .
	 * <p/>
	 * As of SamIam version 3.0, the removal of the current state may produce a
	 * {@code NullPointerException}. Use {@code BeliefNetwork.clone()} or
	 * {@code BeliefNetwork.deepClone()} if this method needs to be called
	 * more than once on the same method to get around this.
	 * 
	 * @param bn
	 *            A SamIam Bayesian network, desired to be put into state
	 * @param evidence
	 *            The state to put {@code bn} into
	 * @return A new {@code BeliefNetwork} with each variable node set to the
	 *         state indicated in the corresponding {@code evidence} attribute
	 *         (excepting the node corresponding to {@code evidence}'s class
	 *         attribute)
	 * @throws StateNotFoundException
	 *             If the instances are not compatible with the Bayesian network
	 *             (they specify states for the nodes that are not among those
	 *             nodes' possible values)
	 * @since version 0.04 2016-04-21
	 */
	public static void loadEvidenceFromWeka(BeliefNetwork bn, Instance evidence) throws StateNotFoundException{
		Map<FiniteVariable, Object> obs = new HashMap<FiniteVariable, Object>(evidence.numValues()-1);
		for(int i=evidence.numValues()-1; i>=0; i--){
			if(i != evidence.classIndex() && !evidence.isMissing(i)){
				FiniteVariable var = (FiniteVariable) bn.forID( evidence.attribute(i).name() );
				String state = evidence.stringValue(i);
				obs.put(var, var.instance(state));
			}
		}
			
		bn.getEvidenceController().setObservations( obs );
		
	}
	
	/**
	 * Computes the marginal probabilities for the given instance using the
	 * Shenoy-Shafer belief propagation algorithm.
	 * <p/>
	 * This code is heavily based on the sample code by Keith Cascio
	 * 
	 * @param bn
	 *            The bayesian network to use for the estimation
	 * @param evidence
	 *            The state the bayesian network should be in when the marginal
	 *            probability is computed
	 * @return A {@code edu.ucla.belief.Table} containing the result
	 * @throws StateNotFoundException
	 *             If the instances are not compatible with the Bayesian network
	 *             (they specify states for the nodes that are not among those
	 *             nodes' possible values)
	 * @since version 0.04 2016-04-21
	 */
	//TODO: define an interface where the marginal function returns
	//the probabilities as a map or list
	public static Table shenoyShaferMarginals(BeliefNetwork bn,
			Instance evidence) throws StateNotFoundException {
		/* Create the Dynamator(edu.ucla.belief.inference.SynchronizedInferenceEngine$SynchronizedPartialDerivativeEngine). */
		edu.ucla.belief.inference.JEngineGenerator dynamator = new edu.ucla.belief.inference.JEngineGenerator();

		/* Edit settings. */
		edu.ucla.belief.inference.JoinTreeSettings settings = dynamator.getSettings( (PropertySuperintendent)bn, true );
		/*
		  Define the elimination order heuristic used to create the join tree, one of:
		    MIN_FILL (Min Fill), MIN_SIZE (Min Size)
		*/
		settings.setEliminationHeuristic( EliminationHeuristic.MIN_FILL );

		/* Create the InferenceEngine. */
		InferenceEngine engine = dynamator.manufactureInferenceEngine( bn );

	    /* Set evidence. */
		loadEvidenceFromWeka(bn, evidence);

	    /* Calculate Pr(e). TODO: do we need this? Check results with and without against manually setting the state in SamIam*/
	    double pE = engine.probability();
	    //System.out.println( "Pr(e): " + pE + "\n");

		/* Define the set of variables for which we want marginal probabilities, by id. */
		FiniteVariable varMarginal = (FiniteVariable) bn.forID( evidence.classAttribute().name() );
		Table answer = engine.conditional( varMarginal );
		
	    /* Clean up to avoid memory leaks. */
	    engine.die();
	    
		return answer;
	}

	/**Loads a Bayesian network from an XML BIF file into a SamIam {@code BeliefNetwork}
	 * 
	 * @param filename
	 *            The path of an XML BIF file to open
	 * @return a {@code weka.classifiers.bayes.BayesNet} object with the
	 *         structure in the file
	 * @throws Exception
	 *             If the file could not be read correctly
	 * @since version 0.04 2016-04-21
	 */
	//TODO: create new class (Tentatively named BifDowngrade or BiffToSamIam)
	public static BeliefNetwork loadSamiamBayes(String filename) throws Exception {
		String[] notes = new String[0];
		File f = new File(filename);
		BeliefNetwork bn = new XmlbifParser().beliefNetwork( f, new NodeLinearTask("default", new SkimmerEstimator(f), 1, notes) );
		return bn;
	}

	/**
	 * Evaluates the given Bayesian network on the given test data using Weka's
	 * built-in evaluator. Note that this evaluator does not support belief
	 * propagation (i.e. if the network is a chain of length 3, and the target
	 * node is <i>v<sub>3</sub></i> and the value of its immediate parent
	 * <i>v<sub>2</sub></i> is missing, rather than check <i>v<sub>1</sub></i>
	 * to predict <i>v<sub>3</sub></i>, the evaluator merely assumes that
	 * <i>v<sub>2</sub> is its default value&mdash;the value of
	 * <i>v<sub>2</sub></i> with the highest frequency in the training data
	 * irrespective of the values of its adjacent nodes)
	 * 
	 * @param bn
	 *            A trained Bayesian network to be tested
	 * @param trainset
	 *            The data the Bayesian network was trained with
	 * @param trainset
	 *            The data to test the Bayesian network with
	 * @return A string containing the results of the evaluation (see
	 *         {@code weka.classifiers.evaluation.Evaluation.toSummaryString})
	 * @throws Exception
	 *             if the Bayesian network is corrupted
	 * @since 0.03 2016-04-20
	 */
	//TODO: iterate over all attributes to test the ability predicting all variables.
	//TODO: Define an interface where an evaluation function returns the
	//confusion table and define methods to let the callers deal with it
	public static String wekaEvaluation(EditableBayesNet bn,
			Instances trainset, Instances testset)
			throws Exception {
		Evaluation e = new Evaluation(trainset);
		e.evaluateModel(bn, testset);
		return e.toSummaryString(true);
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
		EditableBayesNet wekaBayes = BifUpdate.loadBayesNet(args[1]);
	
		// filter out by criterion
		if (args.length >= 4) {
			System.out.println("Filtering by " + data.attribute(0).name()
					+ " equal to " + args[3] + "...");
			data = Trainer.filterByCriterion(args[3], data, 0);
		}
	
		System.out.println("Conforming data to network...");
		if (args.length >= 5)
			data = Trainer.conformToNetwork(data, wekaBayes,
					Boolean.getBoolean(args[4]));
		else
			data = Trainer.conformToNetwork(data, wekaBayes, false);
	
		float trainSize = 0.85f;
	
		// TODO move to function
		assert trainSize > 0;
		assert trainSize < 1;
		
		Map<String, DoubleSummaryStatistics> results = new HashMap<String, DoubleSummaryStatistics>(data.numAttributes());
		int folds = Math.round(1 / (1 - trainSize)) * data.attribute(0).numValues(); //assuming all attributes have the same number of values
		Map<Instances, Instances> Trainset = randomSplit(data, trainSize, folds);
		Iterator<Map.Entry<Instances,Instances>> trains = Trainset.entrySet().iterator();
		@SuppressWarnings("deprecation")
		String date = new java.util.Date().toLocaleString();
		System.out.println("\nRESULTS\nat " + date + "\n-------");
		while (trains.hasNext()) {
			long t = System.nanoTime();
			Map.Entry<Instances,Instances> current = trains.next();
			Instances training = current.getKey();
			Trainer.trainToFile(wekaBayes, training, args[2]);
			String filename = args[2];
			BeliefNetwork bn1 = loadSamiamBayes(filename);
			
			//find accuracy for all classes
			Instances testing = current.getValue();
			for(int k=0; k<testing.numAttributes(); k++){
				testing.setClassIndex(k);
				try{
					double accuracy = accuracy(bn1, testing);
					DoubleSummaryStatistics subtotal;
					if(results.containsKey(testing.attribute(k).name()))
						subtotal = results.get(testing.attribute(k).name());
					else{
						subtotal = new DoubleSummaryStatistics();
					}
					subtotal.accept(accuracy);
					results.put(testing.attribute(k).name(), subtotal);
				}catch(ArithmeticException e){
					//Nothing to compare against. Too bad! Maybe the next split will do better
					//Omit adding
				}
			}
			
		    //String summary = wekaEvaluation(bn, current, Trainset.get(current));
			//System.out.println(summary);
			folds--;
			System.out.println("Processed split in "+(double)(System.nanoTime()-t)/TimeUnit.SECONDS.toNanos(1)+" seconds; "+folds+" folds remain");
		}
		Iterator<Map.Entry<String, DoubleSummaryStatistics>> I = results.entrySet().iterator();
		while(I.hasNext()){
			Map.Entry<String, DoubleSummaryStatistics> e = I.next();
			System.out.println(e.getKey()+": min "+e.getValue().getMin()+"; max "+e.getValue().getMax()+"; average "+e.getValue().getAverage());
		}
	}

	/**Computes the accuracy the Bayesian network has on the given test data.
	 * That is, it attempts to predict the value of the class attribute,
	 * given the rest of the data in the instance, and returns how many
	 * predictions matched the value given in their instance.
	 * <p/>
	 * The function uses the the attribute set as class to know what to
	 * estimate. The class must be set by the caller.
	 * <p/>
	 * The algorithm used is the Shenoy-Shafer belief
	 * propagation algorithm 
	 * 
	 * @param bn The Bayesian network to be evaluated with this data 
	 * @param testData A set of instances, distinct from the ones used to learn
	 * the network's conditional probabilities
	 * 
	 * @return a number between 0 and 1, specifying what ratio, of the values
	 * in {@code testData.instance(i).classValue()}, were correctly predicted
	 * 
	 * @throws StateNotFoundException
	 *             If the instances are not compatible with the Bayesian network
	 *             (they specify states for the nodes that are not among those
	 *             nodes' possible values)
	 * @throws ArithmeticException
	 *             If all the values for the class in {@code testData} are missing
	 * @throws UnassignedClassException
	 *             If {@code testData}'s class attribute is not set
	 * @see {@link #shenoyShaferMarginals(BeliefNetwork, Instance)}
	 * @since 0.6 2016-04-22
	 */
	public static double accuracy(BeliefNetwork bn, Instances testData)
			throws StateNotFoundException, ArithmeticException, UnassignedClassException {
		//fail fast
		if(testData.attributeStats(testData.classIndex()).missingCount == testData.numInstances())
			throw new ArithmeticException("Cannot compute how many of the known values are correctly predicted "
					+ "if there are no known values (cannot divide by zero)");
		double matches = 0;
		int knownResults=0;
		for(int i=0; i<testData.numInstances(); i++){
			Instance evidence = testData.instance(i);
			Table answer = shenoyShaferMarginals(bn.deepClone(), evidence);
		    //System.out.println( i+":"+Trainer.getAttributeNames(current).toString() +"\n" + evidence+"\ngives:\n"+answer.tableString() );
			double[] marginalProbs = answer.dataclone();
			
			//Get the prediction and see if it was what was expected
			//TODO move this to a method (the check below only makes sense there)
		    if(evidence.classAttribute().numValues() != marginalProbs.length)
				throw new IllegalArgumentException("The number of possible values in the evidence does not match the number of values in marginalProbs. Use evidence.classAttribute().enumerateValues() to check what these values are.");
			int max = -1;
			for(int j=marginalProbs.length-1; j>=0; j--)
				if(max == -1 || marginalProbs[j]>marginalProbs[max])
					max=j;
				//TODO account for equal (or very close) probabilities
			
			if(!evidence.classIsMissing()){
				knownResults++;
				if(max == evidence.classValue()){
					matches++;
				}
			}
		}
		assert knownResults>0; //otherwise the fail-fast ArithmeticException above, failed
		return matches/knownResults;
	}

}
