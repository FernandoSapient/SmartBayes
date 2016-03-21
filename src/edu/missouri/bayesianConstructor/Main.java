/**
 * 
 */
package edu.missouri.bayesianConstructor;

import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
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
import edu.ucla.belief.FiniteVariable;
import edu.ucla.belief.FiniteVariableImpl;
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
 * @version 0.05&frac12; 2016-03-21
 * @since {@link bayesianConstructor} version 0.03 2016-03-18
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
	 * &radic;((&sum;(<i>Y</i>&minus;<i>Y</i>&#x305) &minus;
	 * (&sum;(<i>X</i>&minus
	 * ;<i>X</i>&#x305)(<i>Y</i>&minus;<i>Y</i>&#x305))&sup2; /
	 * &sum;((<i>X</i>&minus;<i>X</i>&#x305)&sup2;)) / (n-2)) STE returns values
	 * in the (0, <i>Y</i>&#x305) interval where <i>Y</i>&#x305 is the
	 * arithmetic sample means of <i>Y</i>.
	 * <p?>
	 * If a value is {@code null}; it, and the corresponding value in the other
	 * list, are ignored.
	 * 
	 * @param X
	 *            The list of values for the variable thought to be independent
	 * @param Y
	 *            The list of values for the variable thought to be independent
	 * @return the dependency score, a number between 0 and <i>Y</i>&#x305
	 * @throws IllegalArgumentException
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
	 * negative dependency. The forward dependency is defined as
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
	 * @deprecated since version 0.05 (2016-03-20). A graph of
	 *             nodes serves no other purpose than to create a
	 *             {@code edu.ucla.belief.BeliefNetwork}; however, the
	 *             graphs returned by this function are not fully supported
	 *             by {@code BeliefNetwork} (CPT tables do not get initialized
	 *             properly). Use {@link #graphToNetwork(DirectedGraph, String[])
	 *             for this purpose instead.
	 */
	//Note Hugin nodes are the only instantiable subclass of StandardNode
	@Deprecated
	@SuppressWarnings({ "unchecked" })
	public static DirectedGraph graphToNodeGraph(DirectedGraph g, String[] values) {
		DirectedGraph g_prime = (DirectedGraph) g.clone();
		Iterator<String> V = (Iterator<String>) (g.vertices().iterator());
		while (V.hasNext()) {
			String v = V.next();
			g_prime.replaceVertex(v, new HuginNodeImpl(new FiniteVariableImpl(
					v, values)));// this begs the question: if Hugin Nodes
									// contain CPTs, where do they get them
									// from?
		}
		return g_prime;
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
	 * @return a {@code BeliefNetwork} with one node and edge for every node and
	 *         edge in the given graph {@code g}
	 * @since 0.05 2016-03-20
	 */
	@SuppressWarnings("unchecked")
	//TODO: Make this more efficient by storing pending edges hashed by their second vertex.
	//		That way, they can all be added as soon as the missing vertex is created
	public static BeliefNetwork graphToNetwork(DirectedGraph g, String[] values) {
		BeliefNetwork out = new BeliefNetworkImpl();
		boolean added;//used to check correctness
		Set<String> vertices = (Set<String>)g.vertices();
		Map<String, FiniteVariable> representations = new HashMap<String,FiniteVariable>(vertices.size()); //allows working around FiniteVariable's lack of an "equals" method
		Set<Edge> pending = new HashSet<Edge>(); //Stores edges that couldn't be added due to hash ordering
		FiniteVariable v_rep, d_rep;
		Iterator<String> V = vertices.iterator();
		while (V.hasNext()) {
			String v = V.next();
			if(representations.containsKey(v)){
				v_rep = representations.get(v); 
			}else{
				v_rep = new FiniteVariableImpl(v, values);
				representations.put(v, v_rep);
			}
			added = out.addVariable(v_rep, true);
			assert added; // if this fails, g had two nodes with the same name,
							// which is impossible
			Iterator<String> dependents = (Iterator<String>) (g.outGoing(v)
					.iterator());
			while (dependents.hasNext()) {
				String d = dependents.next();
				if(representations.containsKey(d)){
					d_rep = representations.get(d); 
				}else{
					d_rep = new FiniteVariableImpl(d, values);
					representations.put(d, d_rep);
				}
				if(out.contains(d_rep)){
					added = out.addEdge(v_rep, d_rep);
				}else{
					added = pending.add(new Edge(v_rep, d_rep));
				}
				if (!added)
					throw new IllegalArgumentException(
							"Could not add edge from "+d+" to "+v+". Please make sure "
								+"the graph provided is not a multigraph.");
				assert out.isAcyclic();
			}
		}
		Set<FiniteVariable> addedVertices = (Set<FiniteVariable>)(out.vertices());
		assert addedVertices.size() == out.vertices().size();
		assert g.numEdges() > out.numEdges();
		
		Iterator<Edge> missingEdges = pending.iterator();
		while(missingEdges.hasNext()){
			Edge e = missingEdges.next();
			assert addedVertices.contains(e.v1());
			assert addedVertices.contains(e.v2());
			added = out.addEdge(e.v1(), e.v2());
			if (!added)
				throw new IllegalArgumentException(
						"Could not add edge from "+e.v1()+" to "+e.v2()+". Please make sure "
							+"the graph provided is not a multigraph.");
			assert out.isAcyclic();
		}
		assert g.numEdges() == out.numEdges();
		return out;
	}

	/**
	 * Sample main program
	 * 
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err
					.println("Usage: java Main <data file> [filter criterion]");
			return;
		}

		// Read data
		CSVReader reader = new CSVReader(new FileReader(args[0]));
		List<String> titles = Arrays.asList(reader.readNext()); // we will need
																// to check
																// "contains" on
																// this later
		assert titles != null;
		int cols = titles.size();
		System.err.println(cols + " variable columns found. Filtering by "
				+ titles.get(0) + " equal to " + args[1]);

		// initialize data array
		@SuppressWarnings("unchecked")
		Vector<Double>[] data = (Vector<Double>[]) new Vector[cols];
		for (int i = 0; i < cols; i++)
			data[i] = new Vector<Double>();

		// load file into data array
		String[] nextLine;
		int rows = 0;
		while ((nextLine = reader.readNext()) != null) {
			assert nextLine.length == cols;
			if (args.length >= 1 && args[1].equals(nextLine[0])) {
				rows++;
				for (int i = 0; i < cols; i++) {
					if (nextLine[i].isEmpty())
						data[i].addElement(null);
					else
						try {
							data[i].addElement(new Double(nextLine[i]));
						} catch (NumberFormatException e) {
							data[i].addElement(null);
						}
					assert data[i].size() == rows; // all vectors must be same
													// size
				}
			}
		}
		reader.close();
		System.err.println("loaded " + rows + " rows of data");

		// expected names
		String primary = "Labor force with primary education (% of total) [SL.TLF.PRIM.ZS]", secondary = "Labor force with secondary education (% of total) [SL.TLF.SECO.ZS]", tertiary = "Labor force with tertiary education (% of total) [SL.TLF.TERT.ZS]", journal = "Scientific and technical journal articles [IP.JRN.ARTC.SC]", trademark = "Trademark applications, total [IP.TMK.TOTL]", government = "General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]", foreignAid = "Net official development assistance and official aid received (constant 2012 US$) [DT.ODA.ALLD.KD]", agriculture = "Agriculture, value added (% of GDP) [NV.AGR.TOTL.ZS]", industry = "Industry, value added (% of GDP) [NV.IND.TOTL.ZS]", manufacture = "Manufacturing, value added (% of GDP) [NV.IND.MANF.ZS]", services = "Services, etc., value added (% of GDP) [NV.SRV.TETC.ZS]", unemployed = "Unemployment, total (% of total labor force) [SL.UEM.TOTL.ZS]", growth = "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]", PPP = "GDP per capita, PPP (constant 2011 international $) [NY.GDP.PCAP.PP.KD]";

		// set categories
		List<List<Double>> education = Arrays.asList(
				data[titles.indexOf(primary)], data[titles.indexOf(secondary)],
				data[titles.indexOf(tertiary)]), innovation = Arrays.asList(
				data[titles.indexOf(journal)], data[titles.indexOf(trademark)],
				data[titles.indexOf(government)],
				data[titles.indexOf(foreignAid)]), production = Arrays.asList(
				data[titles.indexOf(agriculture)],
				data[titles.indexOf(industry)],
				data[titles.indexOf(manufacture)],
				data[titles.indexOf(services)],
				data[titles.indexOf(unemployed)]), economic = Arrays.asList(
				data[titles.indexOf(growth)], data[titles.indexOf(PPP)]);

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Education", Arrays.asList(primary, secondary, tertiary));
		m.addLayer("Innovation",
				Arrays.asList(journal, trademark, government, foreignAid));
		m.addLayer("Production", Arrays.asList(agriculture, industry,
				manufacture, services, unemployed));
		m.addLayer("Economic", Arrays.asList(growth, PPP));
		m.addDependency("Education", "Innovation",
				getDependency(education, innovation));
		m.addDependency("Education", "Production",
				getDependency(education, production));
		// m.addDependency("Innovation", "Production", getDependency(innovation,
		// production));
		m.addDependency("Production", "Economic",
				getDependency(production, economic));

		DirectedGraph variableGraph = m.variableDependency(.03);

		// define values
		String[] values = { "high", "med", "low" };

		// convert to bayesian network
		//BeliefNetwork out = new BeliefNetworkImpl(graphToNodeGraph(variableGraph, values)); // Does not seem to be working properly
		BeliefNetwork out = graphToNetwork(variableGraph, values);

		PrintStream f = new PrintStream(new File("testOut.txt"));
		XmlbifWriter w = new XmlbifWriter();
		w.write(out, f);
	}

}
