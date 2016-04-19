package edu.missouri.bayesianConstructor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import edu.ucla.belief.BeliefNetwork;
import edu.ucla.belief.BeliefNetworkImpl;
import edu.ucla.belief.FiniteVariableImpl;
import edu.ucla.belief.io.StandardNode;
import edu.ucla.belief.io.hugin.HuginNodeImpl;
import edu.ucla.belief.io.xmlbif.XmlbifWriter;
import edu.ucla.structure.DirectedGraph;
import edu.ucla.structure.Edge;

import com.opencsv.CSVReader;

/**
 * Provides a sample program that creates a Bayesian Network from economic data
 * and stores it in an XML BIF file. The sample file can be found in the root
 * folder. The functions contained herein are subject to be moved to other
 * classes.
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.13 2016-04-19
 * @since {@code bayesianConstructor} version 0.03 2016-03-18
 */
public class Main {
	/**
	 * Computes the dependency between the two given lists of data. The
	 * dependency is defined as
	 * 1&minus;<i>STE</i>(<i>X</i>&rarr;<i>Y</i>)/<i>Y</i>&#x305;, where STE is
	 * the <a href=
	 * "https://support.office.com/en-us/article/STEYX-function-6ce74b2c-449d-4a6e-b9ac-f9cef5ba48ab"
	 * >standard error of the predicted y-value for each x in a linear
	 * regression</a> as defined by Microsoft:
	 * &radic;((&sum;(<i>Y</i>&minus;<i>Y</i>&#x305;) &minus;
	 * (&sum;(<i>X</i>&minus
	 * ;<i>X</i>&#x305)(<i>Y</i>&minus;<i>Y</i>&#x305;))&sup2; /
	 * &sum;((<i>X</i>&minus;<i>X</i>&#x305)&sup2;)) / (n-2)) STE returns values
	 * in the (0, <i>Y</i>&#x305;) interval where <i>Y</i>&#x305 is the
	 * arithmetic sample means of <i>Y</i>.
	 * <p/>
	 * If a value is {@code null}; it, and the corresponding value in the other
	 * list, are ignored.
	 * 
	 * @param X
	 *            The list of values for the variable thought to be independent
	 * @param Y
	 *            The list of values for the variable thought to be independent
	 * @return the dependency score, a number between 0 and <i>Y</i>&#x305;
	 * @throws IllegalArgumentException
	 *             If {@code X} and {@code Y} are not of equal size
	 * @since 0.01 2016-03-15
	 */
	// TODO: There MUST be a way to make this more efficient (see
	// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
	// )
	public static double dependency(List<Double> X, List<Double> Y)
			throws IllegalArgumentException {
		if (X.size() != Y.size())
			throw new IllegalArgumentException(
					"Both lists must have the same number of elements");
		int n = X.size();

		// total
		double sumX = 0;
		double sumY = 0;
		int nx = 0;
		for (int i = 0; i < n; i++) {
			if (X.get(i) != null && Y.get(i) != null) {
				sumX += X.get(i).doubleValue();
				sumY += Y.get(i).doubleValue();
				nx++;
			}
		}

		double stdNumX = 0;// X's standard deviation numerator (x-avg(x))^2
		double stdNumY = 0;// Y's standard deviation numerator (y-avg(y))^2
		double covNum = 0;// covariance numerator ((x-avg(x))(y-avg(y))^2)
		int processed = 0;
		for (int i = 0; i < n; i++) {
			if (X.get(i) != null && Y.get(i) != null) {
				double xi = X.get(i).doubleValue();
				double yi = Y.get(i).doubleValue();
				stdNumX += Math.pow(xi - sumX / nx, 2);
				stdNumY += Math.pow(yi - sumY / nx, 2);
				covNum += (xi - sumX / nx) * (yi - sumY / nx);
				processed++;
			}
		}
		assert processed == nx;

		double STE = Math.sqrt((stdNumY - Math.pow(covNum, 2) / stdNumX)
				/ (nx - 2));
		double out = 1 - STE / (sumY / nx);
		assert (out >= 0 && out <= 1);
		return out;
	}

	/**
	 * Compute the degree to which {@code dependent} depends on
	 * {@code independent} by subtracting the forward dependency minus the
	 * backward dependency. The forward dependency is defined as
	 * 1&minus;<i>STE</i>(<i>X</i>&rarr;<i>Y</i>)/<i>Y</i>&#x305; while the
	 * backward dependency is defined as
	 * 1&minus;<i>STE</i>(<i>Y</i>&rarr;<i>X</i>)/<i>X</i>&#x305; where <i>X</i>
	 * is the independent data and <i>Y</i> is the dependent variable. The
	 * result should be between -1 and 1 unless the values are less than the
	 * minimum.
	 * <p/>
	 * Only the values whose forward dependency is greater than the minimum are
	 * kept. If a forward dependency comes out less than the minimum, it is
	 * returned as {@code Double.NEGATIVE_INFINITY}. If these values are
	 * desired, {@code minimum} can be safely set to 0.
	 * <p/>
	 * 
	 * @param independent
	 *            the list of values for the variables assumed to be
	 *            independent&mdash;each sublist is assumed to be a different
	 *            variable.
	 * @param dependent
	 *            the list of values for the variables assumed to be
	 *            dependent&mdash;each sublist is assumed to be a different
	 *            variable.
	 * 
	 * @return a {@code independent.size()} by {@code dependent.size()} table of
	 *         {@code Double}s containing, in each position <i>i</i>, <i>j</i>,
	 *         the degree at which {@code dependent.get(j)} depends on
	 *         {@code dependent.get(i)}
	 * 
	 * @since 0.01 2016-03-15
	 */
	public static Double[][] getDependency(List<List<Double>> independent,
			List<List<Double>> dependent, double minimum) {
		int n = independent.size();
		int m = dependent.size();
		Double[][] out = new Double[n][m];

		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++) {
				double x_to_y = dependency(independent.get(i), dependent.get(j));
				if (x_to_y > minimum) {
					double y_to_x = dependency(dependent.get(j),
							independent.get(i));
					out[i][j] = new Double(x_to_y - y_to_x);
					assert (out[i][j].doubleValue() > -1 && out[i][j] < 1);
				} else {
					out[i][j] = Double.NEGATIVE_INFINITY;
				}
			}
		return out;
	}

	/**
	 * Compute the degree to which {@code dependent} depends on
	 * {@code independent} by subtracting the forward dependency minus the
	 * negative dependency. The forward dependency is defined as
	 * 1&minus;<i>STE</i>(<i>X</i>&rarr;<i>Y</i>)/<i>Y</i>&#x305; while the
	 * backward dependency is defined as
	 * 1&minus;<i>STE</i>(<i>Y</i>&rarr;<i>X</i>)/<i>X</i>&#x305; where <i>X</i>
	 * is the independent data and <i>Y</i> is the dependent variable. The
	 * result should be between -1 and 1 unless the values are less than the
	 * 0.5.
	 * <p/>
	 * Only the values whose forward dependency is greater than 0.5 are kept. If
	 * a forward dependency comes out less than 0.5, it is returned as
	 * {@code Double.NEGATIVE_INFINITY}. If these values are desired, use
	 * {@code getDependency(independent, dependent, 0)}
	 * 
	 * @param independent
	 *            the list of values for the variables assumed to be
	 *            independent&mdash;each sublist is assumed to be a different
	 *            variable.
	 * @param dependent
	 *            the list of values for the variables assumed to be
	 *            dependent&mdash;each sublist is assumed to be a different
	 *            variable.
	 * 
	 * @since 0.01 2016-03-15
	 */
	public static Double[][] getDependency(List<List<Double>> independent,
			List<List<Double>> dependent) {
		return getDependency(independent, dependent, 0.5);
	}

	/**
	 * Wraps the name of each vertex in {@code g} in a
	 * {@code edu.ucla.belief.StandardNode} containing the specified
	 * {@code values} as the variables possible finite values.
	 * 
	 * @param g
	 *            the graph to be converted. Vertices must be {@code String}s
	 * @param values
	 *            list of values for the new variables
	 * @return a graph g' were every vertex <i>v</i> now contains
	 *         {@code StandardNode(v,values)}
	 * @throws ClassCastException
	 *             if the vertices in g are not {@code String}s
	 * @since 0.03 2016-03-15
	 * @deprecated since version 0.05 (2016-03-20). A graph of nodes serves no
	 *             other purpose than to create a
	 *             {@code edu.ucla.belief.BeliefNetwork}; however, the graphs
	 *             returned by this function are not fully supported by
	 *             {@code BeliefNetwork} (Edges do not get updated properly).
	 *             Use {@link #graphToNetwork(DirectedGraph, String[]) for this
	 *             purpose instead.
	 */
	@Deprecated
	@SuppressWarnings({ "unchecked" })
	public static DirectedGraph graphToNodeGraph(DirectedGraph g,
			String[] values) {
		DirectedGraph g_prime = (DirectedGraph) g.clone();
		Iterator<String> V = (Iterator<String>) (g.vertices().iterator());
		while (V.hasNext()) {
			String v = V.next();
			g_prime.replaceVertex(v, new HuginNodeImpl(new FiniteVariableImpl(
					v, values))); // NOTE: Hugin nodes are the only instantiable
									// subclass of StandardNode

			// replaceVertex creates null pointers
		}
		return g_prime;
	}

	/** Finds the index of the layer with the most elements */
	private static <T, U> int biggestLayer(Map<T, List<U>> x) {
		Iterator<List<U>> layers = x.values().iterator();
		int out = 0;
		while (layers.hasNext())
			out = Math.max(out, layers.next().size());
		return out;
	}

	/** Finds the index of the layer {@code d} is in, in {@code structure} */
	private static <T, U> int findLayer(Map<T, List<U>> structure, U d) {
		Iterator<List<U>> layers = structure.values().iterator();
		int i = 0;
		while (layers.hasNext()) {
			if (layers.next().contains(d))
				return i;
			i++;
		}
		throw (new IllegalArgumentException(d.toString()
				+ " could not be found in any of the layers"));
	}

	/**
	 * Finds the index of {@code d} within the layer that it's in, in
	 * {@code structure}. To find which layer this is use
	 * {@link #findLayer(Map, Object)}
	 */
	private static <T, U> int findIndex(Map<T, List<U>> structure, U d) {
		Iterator<List<U>> layers = structure.values().iterator();
		while (layers.hasNext()) {
			int out = layers.next().indexOf(d);
			if (out > -1)
				return out;
		}
		throw (new IllegalArgumentException(d.toString()
				+ " could not be found in any of the layers"));
	}

	/**
	 * Finds the number of element of the layer {@code d} is in, in
	 * {@code structure}
	 */
	private static <T, U> int findSize(Map<T, List<U>> structure, U d) {
		Iterator<List<U>> layers = structure.values().iterator();
		while (layers.hasNext()) {
			List<U> L = layers.next();
			if (L.contains(d))
				return L.size();
		}
		throw (new IllegalArgumentException(d.toString()
				+ " could not be found in any of the layers"));
	}

	/**
	 * Converts the given graph {@code g} into a {@code BeliefNetwork}. The
	 * function iterates through all the elements in the graph linearly and
	 * creates a {@code BeliefNetwork} copy as it goes. All nodes in the new
	 * Belief Network are given the same set of possible values: those in the
	 * {@code values} parameter. For this reason a universal set of names for
	 * discretized parameters is encouraged (such as "high", and "low")
	 * 
	 * @param g
	 *            The graph to base the new {@code BeliefNetwork} on. Vertices
	 *            must be {@code String}s
	 * @param values
	 *            The possible values for all nodes
	 * @param structure
	 *            A map showing what nodes should be grouped together when
	 *            plotted. (Each key is a component identifier and each list is
	 *            the nodes in that component.
	 *            {@link DomainKnowledge#layerMap()} can be used).
	 * @param config
	 *            How to dispose the nodes on the canvas. Must be one of the
	 *            {@link NodePlacer} valid configurations
	 * @param angle
	 *            What angle to use for the placement (see {@link NodePlacer})
	 * @return a {@code BeliefNetwork} with one node and edge for every node and
	 *         edge in the given graph {@code g}
	 * @since 0.09 2016-04-08
	 */
	@SuppressWarnings("unchecked")
	// TODO:Make this more efficient by storing pending edges hashed by their
	// second vertex. That way, they can all be added as soon as the missing
	// vertex is created
	// TODO:Move main body to a method that takes a NodePlacer, that way callers
	// can set all the options themselves
	public static BeliefNetwork graphToNetwork(DirectedGraph g,
			String[] values, Map<String, List<String>> structure, char config,
			double angle) {
		BeliefNetwork out = new BeliefNetworkImpl();
		boolean added;// used to check correctness
		Set<String> vertices = (Set<String>) g.vertices();
		Map<String, StandardNode> representations = new HashMap<String, StandardNode>(
				vertices.size()); // allows working around FiniteVariable's lack
									// of an "equals" method
		Set<Edge> pending = new HashSet<Edge>(); // Stores edges that couldn't
													// be added due to hash
													// reordering (both
													// vertices must exist for
													// an edge to be created)
		StandardNode v_rep, d_rep;
		NodePlacer placer = new NodePlacer(structure.keySet().size(),
				biggestLayer(structure), angle, config);
		Iterator<String> V = vertices.iterator();
		while (V.hasNext()) {
			String v = V.next();
			if (representations.containsKey(v)) {
				v_rep = representations.get(v);
			} else {
				v_rep = new HuginNodeImpl(new FiniteVariableImpl(v, values));
				// NOTE: Hugin nodes are the only instantiable subclass of
				// StandardNode
				v_rep.setLocation(placer.position(findLayer(structure, v),
						findIndex(structure, v), structure.keySet().size(),
						findSize(structure, v)));
				representations.put(v, v_rep);
			}
			added = out.addVariable(v_rep, true);
			assert added; // if this fails, g had two nodes with the same name,
							// which is impossible

			Iterator<String> dependents = (Iterator<String>) (g.outGoing(v)
					.iterator());
			while (dependents.hasNext()) {
				String d = dependents.next();
				if (representations.containsKey(d)) {
					d_rep = representations.get(d);
				} else {
					d_rep = new HuginNodeImpl(new FiniteVariableImpl(d, values));
					// NOTE: Hugin nodes are the only instantiable subclass of
					// StandardNode
					d_rep.setLocation(placer.position(findLayer(structure, d),
							findIndex(structure, d), structure.keySet().size(),
							findSize(structure, d)));
					representations.put(d, d_rep);
				}
				if (out.contains(d_rep)) {
					added = out.addEdge(v_rep, d_rep);
				} else {
					added = pending.add(new Edge(v_rep, d_rep));
				}
				if (!added)
					throw new IllegalArgumentException(
							"Could not add edge from " + d + " to " + v
									+ ". Please make sure "
									+ "the graph provided is not a multigraph.");
				assert out.isAcyclic();
			}// end depedendents iteration
		}// end vertices iteration
		Set<StandardNode> addedVertices = (Set<StandardNode>) (out.vertices());
		assert addedVertices.size() == out.vertices().size();
		assert addedVertices.size() == representations.size();

		Iterator<Edge> missingEdges = pending.iterator();
		while (missingEdges.hasNext()) {
			Edge e = missingEdges.next();
			assert addedVertices.contains(e.v1());
			assert addedVertices.contains(e.v2());
			added = out.addEdge(e.v1(), e.v2());
			if (!added)
				throw new IllegalArgumentException("Could not add edge from "
						+ e.v1() + " to " + e.v2() + ". Please make sure "
						+ "the graph provided is not a multigraph.");
			assert out.isAcyclic();
		}
		assert g.numEdges() == out.numEdges();
		return out;
	}

	/**
	 * Converts the given graph {@code g} into a {@code BeliefNetwork} displaced
	 * using a {@link NodePlacer#LAYERED} configuration. All nodes in the new
	 * Belief Network are given the same set of possible values: those in the
	 * {@code values} parameter. For this reason a universal set of names for
	 * discretized parameters is encouraged (such as "high", and "low")
	 * <p/>
	 * This function is shorthand for <a href=#graphToNetwork(DirectedGraph,
	 * String[], Map, char, double)>{@code graphToNetwork}</a>(
	 * {@code g, values, structure,} {@link NodePlacer#LAYERED}, {@code Math.PI}
	 * ) as of version 0.05
	 * 
	 * @param g
	 *            The graph to base the new {@code BeliefNetwork} on. Vertices
	 *            must be {@code String}s
	 * @param values
	 *            The possible values for all nodes
	 * @param structure
	 *            A map showing what nodes should be grouped together when
	 *            plotted. (Each key is a component identifier and each list is
	 *            the nodes in that component {@link DomainKnowledge#layerMap()}
	 *            can be used).
	 * @return a {@code BeliefNetwork} with one node and edge for every node and
	 *         edge in the given graph {@code g}
	 * @since 0.05 2016-03-20
	 */
	public static BeliefNetwork graphToNetwork(DirectedGraph g,
			String[] values, Map<String, List<String>> structure) {
		return Main.graphToNetwork(g, values, structure, NodePlacer.LAYERED,
				Math.PI);
	}

	/**
	 * Converts the given graph {@code g} into a {@code BeliefNetwork} displaced
	 * using a {@link NodePlacer#GEOMETRIC} configuration, placing each node in
	 * a different component All nodes in the new Belief Network are given the
	 * same set of possible values: those in the {@code values} parameter. For
	 * this reason a universal set of names for discretized parameters is
	 * encouraged (such as "high", and "low")
	 * 
	 * @param g
	 *            The graph to base the new {@code BeliefNetwork} on. Vertices
	 *            must be {@code String}s
	 * @param values
	 *            The possible values for all nodes
	 * @return a {@code BeliefNetwork} with one node and edge for every node and
	 *         edge in the given graph {@code g}
	 * @since 0.10 2016-04-18
	 */
	@SuppressWarnings("unchecked")
	public static BeliefNetwork graphToNetwork(DirectedGraph g, String[] values) {
		Set<String> vertices = (Set<String>) g.vertices();
		Map<String, List<String>> structure = new HashMap<String, List<String>>(
				vertices.size());
		Iterator<String> V = vertices.iterator();
		while (V.hasNext()) {
			String v = V.next();
			structure.put(v, Arrays.asList(v));
		}

		return Main.graphToNetwork(g, values, structure, NodePlacer.GEOMETRIC,
				0);

	}

	/**
	 * Sample main program. The program receives a CSV file where the first row
	 * is assumed to be the column names, and in every other row, the first
	 * column is assumed to be a filtering criterion. Which filter criterion to
	 * use can be specified by using the third argument. Any regular expression
	 * may be used in filtering. Filtering is optional.
	 * <p/>
	 * The program assumes all useful data to be real numbers. Anything that
	 * cannot be cast to a real number is represented as {@code null}. For this
	 * reason, the names of columns that do not contain real numbers should not
	 * be included in the domain model.
	 * <p/>
	 * The program generates a Bayesian network with the data and stores it in
	 * XML-BIF format. The resulting file has the filename given in the second
	 * argument.
	 * 
	 * @param args
	 *            An array of length 2-5, where the first position contains the
	 *            name of the file containing the input data, the second
	 *            position contains the name of the file to write to, the third
	 *            position (in the case of length 3) contains a filtering
	 *            regular expression, the fourth (in the case of length 4)
	 *            contains the index this criterion is to be applied to (zero by
	 *            default), and the fifth position (in the case of length 5)
	 *            contains one of the {@link NodePlacer} configuration codes.
	 * @throws IOException
	 *             if the input file could not be read
	 * @throws FileNotFoundException
	 *             if the output file could not be created
	 * @throws NumberFormatException
	 *             if {@code args[3]} is not a valid index because it is not a
	 *             number
	 * @throws ArrayIndexOutOfBoundsException
	 *             if {@code args[3]} is not a valid index because it does not
	 *             fall within the dataset
	 */
	// improve parameter handling. See JCLAP for potential solution
	public static void main(String[] args) throws IOException,
			FileNotFoundException, NumberFormatException {
		if (args.length < 2) {
			System.err
					.println("Usage: java Main <input data file> <output file name>"
							+ " [filter criterion] [filter index] [plot mode]");
			return;
		}

		String filterExp;
		if (args.length < 3)
			filterExp = null;
		else
			filterExp = args[2];

		int filterIndex;
		if (args.length < 4)
			filterIndex = 0;
		else
			filterIndex = Integer.parseInt(args[3]);

		// Read data
		Map<String, List<Double>> data = loadCSVwithFiltering(new FileReader(
				args[0]), filterExp, filterIndex);

		// TODO: read domain knowledge structure from file
		// this will herald in a new major version
		DomainKnowledge m = buildUnescoModel(data);

		DirectedGraph variableGraph = m.variableDependency(.03);

		// define values
		String[] values = { "low", "med", "high" };

		// convert to bayesian network
		BeliefNetwork out;
		// out=newBeliefNetworkImpl(graphToNodeGraph(variableGraph, values));
		if (args.length >= 5)
			out = graphToNetwork(variableGraph, values, m.layerMap(),
					args[4].charAt(0), 0);
		else
			out = graphToNetwork(variableGraph, values, m.layerMap());

		boolean result = networkToFile(out, args[1]);
		if (result)
			System.out.println("File \"" + args[1] + "\" created successfully");
		else
			System.err.println("Could not write file \"" + args[1] + "\".");
	}

	/**
	 * Builds a full {@link DomainKnowledge} model following the structure
	 * proposed in the UNESCO world engineering report. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
	 * 
	 * @param data
	 *            A {@cod Map} representing a column-majoral table, where each
	 *            key is the table's header and the list mapped to is the
	 *            contents of the column with that name.
	 * @return A {@link DomainKnowledge} model with dependnecy tables reflecting
	 *         the relations in the given data
	 * @throws IllegalArgumentException
	 *             If the lists in {@code data} are not all the same size
	 * @since 0.13 2016-04-19
	 */
	private static DomainKnowledge buildUnescoModel(
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

		// Structure
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Economic", Arrays.asList(growth, PPP));
		m.addLayer("Education", Arrays.asList(primary, secondary, tertiary));
		m.addLayer("Innovation",
				Arrays.asList(journal, trademark, government, foreignAid));
		m.addLayer("Production", Arrays.asList(agriculture, industry,
				manufacture, services, unemployed));
		m.addDependency("Education", "Innovation",
				getDependency(education, innovation));
		m.addDependency("Education", "Production",
				getDependency(education, production));
		// m.addDependency("Innovation", "Production", getDependency(innovation,
		// production));
		m.addDependency("Innovation", "Economic",
				Main.getDependency(innovation, economic));
		m.addDependency("Production", "Economic",
				getDependency(production, economic));
		return m;
	}

	/**
	 * Loads the rows from a CSV file that match the given criterion. If all
	 * rows are desired, set {@code filterExp} to {@code null}
	 * 
	 * @param inputFile
	 *            A {@code Reader} that has opened the desired file
	 * @param filterExp
	 *            A regular expression indicating how to filter the
	 *            {@code filterIndex}<sup>th</sup> column
	 * @param filterIndex
	 *            Index of the column desired to be filtered (indices start at
	 *            zero)
	 * @return A {@cod Map} representing a column-majoral table, where each key
	 *         is the table's header and the list mapped to is the contents of
	 *         the column with that name.
	 * @throws IOException
	 *             if at any point it cannot read the next line of the file
	 * @throws IllegalArgumentException
	 *             if the file given contains zero columns in its column header
	 * @throws IllegalStateException
	 *             If not all rows contain the same number of columns
	 * @throws ArrayIndexOutOfBoundsException
	 *             If {@code filterIndex} is greater than the number of columns,
	 *             or if it's negative
	 * @since 0.12 2016-04-19
	 */
	public static Map<String, List<Double>> loadCSVwithFiltering(
			Reader inputFile, String filterExp, int filterIndex)
			throws IOException, IllegalArgumentException,
			IllegalStateException, ArrayIndexOutOfBoundsException {
		CSVReader reader = new CSVReader(inputFile);
		String[] titles = reader.readNext();
		int cols = titles.length;
		if (cols < 1) {
			reader.close();
			throw new IllegalArgumentException("No columns found");
		}

		// initialize data table
		Map<String, List<Double>> data = new HashMap<String, List<Double>>(cols);
		for (int i = 0; i < cols; i++)
			data.put(titles[i], new Vector<Double>());

		// load file into data table
		String[] nextLine;
		int rows = 0;
		while ((nextLine = reader.readNext()) != null) {
			if (nextLine.length != cols) {
				reader.close();
				throw new IllegalStateException(
						"File contains an uneven number of columns in row "
								+ rows);
			}
			if (filterExp == null || filterExp.matches(nextLine[filterIndex])) {
				rows++;
				for (int i = 0; i < cols; i++) {
					if (nextLine[i].isEmpty())
						data.get(titles[i]).add(null);
					else
						try {
							data.get(titles[i]).add(new Double(nextLine[i]));
						} catch (NumberFormatException e) {
							data.get(titles[i]).add(null);
						}
					assert data.get(titles[i]).size() == rows; // all vectors
																// same size
				}
			}
		}
		reader.close();
		return data;
	}

	/**
	 * Loads the rows from a CSV file whose first column match the given
	 * criterion. If all rows are desired, set {@code filterExp} to {@code null}
	 * 
	 * @param inputFile
	 *            A {@code Reader} that has opened the desired file
	 * @param filterExp
	 *            A regular expression indicating how to filter the
	 *            {@code filterIndex}<sup>th</sup> column
	 * @return A {@cod Map} representing a column-majoral table, where each key
	 *         is the table's header and the list mapped to is the contents of
	 *         the column with that name.
	 * @throws IOException
	 *             if at any point it cannot read the next line of the file
	 * @throws IllegalArgumentException
	 *             if the file given contains zero columns in its column header
	 * @throws IllegalStateException
	 *             If not all rows contain the same number of columns
	 * @since 0.12 2016-04-19
	 */
	public static Map<String, List<Double>> loadCSVwithFiltering(
			Reader inputFile, String filterExp) throws IOException,
			IllegalArgumentException, IllegalStateException {
		return loadCSVwithFiltering(inputFile, filterExp, 0);
	}

	/**
	 * Loads a CSV file into a column-majoral representation
	 * 
	 * @param inputFile
	 *            A {@code Reader} that has opened the desired file
	 * 
	 * @return A {@code Map} representing a column-majoral table, where each key
	 *         is the table's header and the list mapped to is the contents of
	 *         the column with that name.
	 * @throws IOException
	 *             if at any point it cannot read the next line of the file
	 * @throws IllegalArgumentException
	 *             if the file given contains zero columns in its column header
	 * @throws IllegalStateException
	 *             If not all rows contain the same number of columns
	 * @since 0.12 2016-04-19
	 */
	public static Map<String, List<Double>> loadCSVwithFiltering(
			Reader inputFile) throws IOException, IllegalArgumentException,
			IllegalStateException {
		return loadCSVwithFiltering(inputFile, null);
	}

	/**
	 * Stores the given {@code BeliefNetwork} in an XML-BIF file with the file
	 * name provided
	 * 
	 * @param bn
	 *            The network to be saved in {@code filename}
	 * @param filename
	 *            The name of the file that will encode {@code bn}
	 * @return {@code True}
	 * @throws FileNotFoundException
	 *             if {@code filename} could not be created
	 * @since 0.10 2016-04-18
	 */
	public static boolean networkToFile(BeliefNetwork bn, String filename)
			throws FileNotFoundException {
		PrintStream f = new PrintStream(new File(filename));
		XmlbifWriter w = new XmlbifWriter();
		boolean result = w.write(bn, f);
		return result;
	}

}
