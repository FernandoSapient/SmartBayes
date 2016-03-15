package edu.missouri.bayesianConstructor;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import edu.ucla.structure.DirectedGraph;
import edu.ucla.structure.DirectedEdge;
import edu.ucla.structure.HashDirectedGraph;

/**
 * Stores information pertinent to the domain knowledge. A domain knowledge is
 * defined by a series of <strong>variables</strong> grouped into
 * <strong>layers</strong>, with the dependence between any two layers being
 * known. These <strong>dependence relations</strong> are stored in a directed
 * acyclic graph and are called the domain knowledge's <em>structure</em>.
 * <p/>
 * The {@code DomainKnowledge} class exists to represent situations where a
 * general relationship between broad concepts (variable layers) is known, but
 * the relationship between the <em>measurable</em> elements of those concepts
 * (the variables themselves) is not. The {@code DomainKnowledge} class
 * therefore assumes that there is some relationship between all variables in
 * the two layers, but that the strength of this relationship is stronger for
 * some pairs of variables than others. The {@code DomainKnowledge} class allows
 * storing these strengths numerically in <strong>dependency tables</strong>,
 * but the values of these strengths must be computed by the caller. These
 * relations can be discretized with the {@link #variableDependency(Double)}
 * function.
 * <p/>
 * A relationship where one layer <i>D</i> is known to depend on another
 * <i>I</i> (denoted <i>I</i>&rarr;<i>D</i>) is called a <strong>dependence
 * relation</strong> and is composed of the name or identifier for the layer
 * depended on <i>I</i>&mdash;called the <strong>independent
 * layer</strong>&mdash;and the layer <i>D</i> that depends on
 * <i>I</i>&mdash;called the <strong>dependent layer</strong>. Note that these
 * names are not absolute; any given layer is allowed to be dependent in one
 * relation, and independent in another.
 * <p/>
 * The constructors initialize the instance with an empty graph. Layers are
 * identified by means of a string referred to as the <strong>layer
 * name</strong>. Layer names must be unique and are case sensitive. Variables
 * are similarly identified by strings, but uniqueness is not enforced (
 * {@link #variableDependency(Double)} will fail, though, if the variable names
 * are not unique). Note however that the order of the variables does matter
 * (for reasons explained below) and should not be altered wantoningly.
 * </p>
 * Layers (vertices in the graph) may be added using the
 * {@link #addLayer(String, List)} method, retrieved with the
 * {@link #getLayer(String)} method, and removed with the
 * {@link #removeLayer(String)} method. The variables a layer is associated with
 * can be modified with {@link #replaceLayer(String, List)}. Note, however, that
 * the class cannot check if the order of the variables has changed, and
 * therefore reinitializes the dependency table when this occurs to prevent
 * bugs. Dependence between layers (graph edges) may be created using the
 * {@link #addDependency(String, String)} method. Note that this method enforces
 * the acyclicity of the graph.
 * <p/>
 * Creation of a dependence relation creates a a dependency table storing the
 * strength of this dependence. The dependency table for any existing relation
 * may be accessed by calling {@link #getDependencyTable(String, String)}. The
 * table can be filled by the caller and then reinserted into the instance using
 * {@link #setDependency(String, String, Double[][])}, which will attempt to
 * ensure the right table is being associated with the right relation. These
 * checks can be avoided by writing directly into the related table through an
 * insecure access to {@link #dependencyTables}. To see if a relationship
 * exists, call {@link #containsDependency(String, String)}. To obtain all the
 * relationships of a node, call {@link #getDependents(String)} and
 * {@link #getIndependents(String)}. Finally, dependence relations can be
 * removed by calling {@link #removeDependency(String, String)}, which will also
 * delete the data in the associated dependency table.
 * <p/>
 * A dependency table is a two-dimensional array containing one row for each
 * variable in the independent layer and one column for each variable in the
 * dependent layer. Thus, for a dependency table <i>T</i> representing the
 * relationship <i>I</i>&rarr;<i>D</i>, the element in the <i>j</i><sup>th</sup>
 * row, <i>k</i><sup>th</sup> column, stores how strongly the
 * <i>k</i><sup>th</sup> variable of <i>D</i> depends on the
 * <i>j</i><sup>th</sup> variable of <i>I</i>&mdash;that is, {@code T[j][k]}
 * stores the strength of {@code I.get(j)}&rarr;{@code D.get(k)}.
 * <p/>
 * Typical usage of this class will see the graph getting built, the dependency
 * tables getting filled and finally a call to
 * {@code #variableDependency(Double)}, at which point the resulting graph
 * between the variables can safely be used instead.
 * <p/>
 * Note that this class allows a layer to have zero variables, either by passing
 * the empty list to it, or by setting its variable list to {@code null}.
 * Consequently, if a dependency is created involving a layer with zero
 * variables, the associated dependency table will have one of its dimensions
 * set to zero. This is intended, to allow for "placeholder" layers (i.e. some
 * future caller may need to compute) the list of variables after creating the
 * layer).
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 1.01 2016-03-14
 * @since {@link bayesianConstructor} version 0.01
 */
// TODO: let user specify type for the layer and variable identifiers <generics>
public class DomainKnowledge {
	/**
	 * Stores the dependence relations between the layers; as such, the graph
	 * has one vertex for every layer and one edge for every relation.
	 */
	protected DirectedGraph layerStructure;

	/**
	 * Stores the names of the variables associated with each layer, indexed by
	 * layer name. Note that layer names are case sensitive
	 */
	protected Map<String, List<String>> layerVariables;

	/**
	 * Stores the dependency tables indexed by relation. To access the desired
	 * dependency table, create a new {@code DirectedEdge} object <i>E</i> and
	 * use {@code dependencyTables.get(E)}. (For instance, to access the
	 * dependency table of <i>I</i>&rarr;<i>D</i>, call
	 * {@code dependencyTables.get(new DirectedEdge(I,D))})
	 */
	public Map<DirectedEdge, Double[][]> dependencyTables;

	/**
	 * Checks if the number of vertices in {@code #layerStructure} matches the
	 * number of layer names in {@code #layerVariables}. This check should be
	 * performed every time the layers are changed.
	 * 
	 * @return {@code true} if there are exactly as many nodes in the graph as
	 *         there are layers; {@code false} otherwise
	 */
	protected boolean layerInvariant() {
		return this.layerStructure.vertices().equals(
				this.layerVariables.keySet());
	}

	/**
	 * Checks if there is a dependency table for each relationship This check
	 * should be performed every time a relationship is added or removed.
	 * 
	 * @return {@code true} if there are exactly as many dependency tables as
	 *         there are dependency relationships; {@code false} otherwise
	 */
	// TODO: perform a stronger check, identifying which edge is missing
	protected boolean dependencyInvariant() {
		return this.layerStructure.numEdges() == this.dependencyTables.size();
	}

	/**
	 * Checks if the dependency tables are the right size. This check should be
	 * performed every time a dependency table is initialized or replaced
	 * 
	 * @return {@code true} if each dependency table has exactly as many rows as
	 *         there are variables in the independent layer, and exactly as many
	 *         columns as there are variables in the dependent layer, of the
	 *         corresponding dependency relationship; {@code false} otherwise.
	 */
	@SuppressWarnings("unchecked")
	protected boolean variableInvariant() {
		Iterator<String> V = (Iterator<String>) (this.layerStructure.vertices()
				.iterator());
		while (V.hasNext()) {
			String v = V.next();
			int rows = this.layerVariables.get(v) == null ? 0
					: this.layerVariables.get(v).size();
			Iterator<String> E = (Iterator<String>) (this.layerStructure
					.outGoing(v).iterator());
			while (E.hasNext()) {
				String e = E.next();
				int cols = this.layerVariables.get(e) == null ? 0
						: this.layerVariables.get(e).size();
				Double[][] table = this.dependencyTables.get(new DirectedEdge(
						v, e));
				if (table.length != rows)
					return false;
				for (int i = 0; i < rows; i++)
					if (table[i].length != cols)
						return false;
			}
		}
		return true;
	}

	/**
	 * Obtains the total number of elements that would result from adding the
	 * given value, plus all the numbers less than it. This is especially useful
	 * for predicting the minimum number of connections we can expect to occur
	 * between <i>n</i> variables
	 * 
	 * @param n
	 *            the number of elements to add
	 * @return &lfloor;n&times;(n-1)&divide;2&rfloor;
	 */
	protected int t(int n) {
		return Math.floorDiv(n * (n - 1), 2);
	}

	/**
	 * Shared code for constructors (see {@link #DomainKnowledge(int)})
	 * 
	 * @param layers
	 *            initial capacity
	 */
	private void construct(int layers) {
		this.layerStructure = new HashDirectedGraph(layers);
		this.layerVariables = new HashMap<String, List<String>>(layers, 1);
		this.dependencyTables = new HashMap<DirectedEdge, Double[][]>(t(layers));

		// sanity check
		assert this.layerInvariant();
	}

	/**
	 * Creates a new Domain Knowledge instance with an initial capacity for the
	 * specified number of layers. The capacity will grow dynamically if more
	 * layers are added
	 * 
	 * @param layers
	 *            the number of layers to be allocated as the initial capacity
	 */
	public DomainKnowledge(int layers) {
		this.construct(layers);
	}

	/**
	 * Creates a new Domain Knowledge instance with an initial capacity for the
	 * default number of layers (3). The capacity will grow dynamically if more
	 * layers are added
	 */
	public DomainKnowledge() {
		this.construct(3);
	}

	/**
	 * Adds a layer to the current instance composed of the given name and list
	 * of variables. The name for the layer must be unique.
	 * 
	 * @param layerName
	 *            the name by which to identify the layer
	 * @param layerVariables
	 *            the new list of variables for this layer
	 * 
	 * @throws IllegalArgumentException
	 *             if a layer with that name already exists
	 */
	public void addLayer(String layerName, List<String> layerVariables)
			throws IllegalArgumentException {
		if (this.layerVariables.containsKey(layerName))
			throw new IllegalArgumentException(
					layerName
							+ " already exists in the specified DomainKnowledge instance. "
							+ "Use removeLayer or replaceLayer to change an existing layer.");
		this.layerVariables.put(layerName, layerVariables);
		this.layerStructure.addVertex(layerName);

		// sanity check
		assert this.layerInvariant();
	}

	/**
	 * Indicates whether a layer with the specified name exists
	 * 
	 * @param layerName
	 *            whose existence in this domain knowledge instance is to be
	 *            tested
	 * @return {@code true} if the instance contains a layer with that name;
	 *         {@code false} otherwise
	 */
	public boolean containsLayer(String layerName) {
		return this.layerVariables.containsKey(layerName);
	}

	/**
	 * Gets the list of variables stored in the given layer, or {@code null} if
	 * no layer with that name exists
	 * 
	 * @param layerName
	 *            The layer whose variables are to be returned
	 * @return the list of variables in the specified layer, or {@code null} if
	 *         this domain knowledge instance does not contain a layer with that
	 *         name
	 */
	public List<String> getLayer(String layerName) {
		return this.layerVariables.get(layerName);
	}

	/**
	 * Removes the layer with the given identifier from the instance. Note that
	 * removal will delete all related dependencies as well
	 * 
	 * @param layerName
	 *            the name of the layer to be removed
	 * 
	 * @return {@code true} if a layer was removed
	 */
	@SuppressWarnings("unchecked")
	public boolean removeLayer(String layerName) {
		List<String> V = this.layerVariables.remove(layerName);

		// remove dependency tables for edges as well
		Iterator<String> I = (Iterator<String>) (this.layerStructure
				.inComing(layerName).iterator());
		while (I.hasNext())
			this.dependencyTables.remove(new DirectedEdge(I.next(), layerName));

		I = (Iterator<String>) (this.layerStructure.outGoing(layerName)
				.iterator());
		while (I.hasNext())
			this.dependencyTables.remove(new DirectedEdge(layerName, I.next()));

		// finally remove vertex from graph (HashGraph, due to its design, will
		// remove the edges automatically)
		boolean out = this.layerStructure.removeVertex(layerName);

		// sanity checks
		assert (out == false ? V == null : true);
		assert this.layerInvariant();

		return out;
	}

	/**
	 * Gets the set of layer names. The set is backed by a map.
	 * 
	 * @return a {@code Set} of {@code Strings} where each string is a layer
	 *         name
	 */
	public Set<String> layerSet() {
		return this.layerVariables.keySet();
	}

	/**
	 * initializes the dependency table for the relation between the two
	 * indicated notes. Note that no cells are initialized
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @return the previously-stored dependency table, or null if no previous
	 *         table exists
	 * @throws NullPointerException
	 *             if either one of the layers does not exist
	 */
	private Double[][] initDependencyTable(String independent, String dependent) {
		int indVar = 0;
		int depVar = 0;
		List<String> vars = this.layerVariables.get(independent);
		if (vars != null)
			indVar = vars.size();

		vars = this.layerVariables.get(dependent);
		if (vars != null)
			depVar = vars.size();

		return this.dependencyTables.put(new DirectedEdge(independent,
				dependent), new Double[indVar][depVar]);
	}

	/**
	 * Checks if the layer exists and throws an exception if it doesn't
	 * 
	 * @param layerName
	 *            name of the layer to check
	 * @throws IllegalArgumentException
	 *             if the layer doesn't exist
	 */
	private void layerMustExist(String layerName)
			throws IllegalArgumentException {
		if (!this.layerVariables.containsKey(layerName))
			throw new IllegalArgumentException(
					layerName
							+ " does not exist in the specified DomainKnowledge instance. "
							+ "Use addLayer to add a new layer.");
	}

	/**
	 * Checks if the dependency exists and throws an exception if it doesn't
	 * 
	 * @param independent
	 *            name of the layer that {@code dependent} must depend on
	 * @param dependent
	 *            name of the layer that must depend on {@code independent}
	 * @throws IllegalStateException
	 *             if the layer doesn't exist
	 */
	private void dependencyCannotExist(String independent, String dependent)
			throws IllegalStateException {
		if (this.layerStructure.containsEdge(independent, dependent))
			throw new IllegalStateException(
					"A dependency relationship between "
							+ independent
							+ " and "
							+ dependent
							+ " already exists in the specified DomainKnowledge instance. "
							+ "use removeDependency or setDependency to modify it.");
	}

	/**
	 * Checks if the given edge creates a cycle and if so, removes it and throws
	 * an exception
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @throws SecurityException
	 *             if adding the edge results in a cycle being created
	 */
	private void preserveAcyclicity(String independent, String dependent)
			throws SecurityException {
		if (!this.layerStructure.isAcyclic()) {
			// roll back
			this.dependencyTables.remove(new DirectedEdge(independent,
					dependent));
			this.layerStructure.removeEdge(independent, dependent);

			// sanity check
			assert this.dependencyInvariant();
			assert this.variableInvariant();

			// announce exception
			throw new SecurityException(
					"Dependency "
							+ independent
							+ "->"
							+ dependent
							+ " could not be created as it would cause the existing dependencies "
							+ "to form a cycle");
		}
	}

	/**
	 * Adds a dependency relationship from the independent layer specified, to
	 * the dependent layer specified initialized with an empty table in
	 * {@link #dependencyTables}. Note that this does not initialize any of its
	 * cells.
	 * <p/>
	 * The method requires the new dependency to keep the graph acyclic and will
	 * roll back any changes if this is not the case.
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @return {@code true} if a dependency was created
	 * @throws IllegalArgumentException
	 *             if either one of the layers does not exist
	 * @throws IllegalStateException
	 *             if a relationship between the layers already exists
	 * @throws SecurityException
	 *             if adding the edge results in a cycle being created
	 */
	public boolean addDependency(String independent, String dependent)
			throws IllegalArgumentException, IllegalStateException,
			SecurityException {
		this.layerMustExist(independent);
		this.layerMustExist(dependent);
		this.dependencyCannotExist(independent, dependent);

		Double[][] old = initDependencyTable(independent, dependent);
		assert old == null;
		boolean out = this.layerStructure.addEdge(independent, dependent);

		this.preserveAcyclicity(independent, dependent);

		// sanity check
		assert this.dependencyInvariant();
		assert this.variableInvariant();

		return out;
	}

	/**
	 * Checks if the given dependency table is the right size for the
	 * {@code independent}&rarr;{@code dependent} relation
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @param dependencyTable
	 *            the new values for the dependency table of the relationship
	 *            between {@code independent} and {@code dependent}
	 * @throws IllegalArgumentException
	 *             if the dependency table provided does not have enough entries
	 *             for each dimension, or if the rows of {@code dependencyTable}
	 *             are of different length
	 */
	private void rightSize(String independent, String dependent,
			Double[][] dependencyTable) throws IllegalArgumentException {
		// check all rows are same length
		int rows = dependencyTable.length;
		int cols = dependencyTable.length > 0 ? dependencyTable[0].length : 0;
		for (int i = 1; i < rows; i++)
			if (dependencyTable[i].length != cols)
				throw new IllegalArgumentException(
						"dependencyTable is not square (row 0 has " + cols
								+ " columns, " + "but row " + i + " has "
								+ dependencyTable[i].length + " columns)");

		// check if it's right for this relationship
		DirectedEdge id = new DirectedEdge(independent, dependent);
		if (rows != this.dependencyTables.get(id).length)
			throw new IllegalArgumentException(
					"dependencyTable has the wrong number of rows (layer "
							+ independent + " has "
							+ this.dependencyTables.get(id).length
							+ " variables, but the dependencyTable is for "
							+ rows + " variables)");
		if (rows > 0 && cols != this.dependencyTables.get(id)[0].length)
			throw new IllegalArgumentException(
					"dependencyTable has the wrong number of rows (layer "
							+ dependent + " has "
							+ this.dependencyTables.get(id)[0].length
							+ " variables, but the dependencyTable is for "
							+ cols + " variables)");
	}

	/**
	 * Adds a dependency relationship from the independent layer specified, to
	 * the dependent layer specified initialized with the specified
	 * {@code dependencyTable}.
	 * <p/>
	 * The method requires the new dependency to keep the graph acyclic and will
	 * roll back any changes if this is not the case.
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @param dependencyTable
	 *            the new values for the dependency table of the relationship
	 *            between {@code independent} and {@code dependent}
	 * @return {@code true} if a dependency was created
	 * @throws IllegalArgumentException
	 *             if either one of the layers does not exist, if the dependency
	 *             table provided does not have enough entries for each
	 *             dimension, or if the rows of {@code dependencyTable} are of
	 *             different length
	 * @throws IllegalStateException
	 *             if a relationship between the layers already exists
	 * @throws SecurityException
	 *             if adding the edge results in a cycle being created
	 */
	public boolean addDependency(String independent, String dependent,
			Double[][] dependencyTable) throws IllegalArgumentException,
			IllegalStateException, SecurityException {
		this.layerMustExist(independent);
		this.layerMustExist(dependent);
		this.dependencyCannotExist(independent, dependent);
		this.rightSize(independent, dependent, dependencyTable);

		Double[][] old = this.dependencyTables.put(new DirectedEdge(
				independent, dependent), dependencyTable);
		assert old == null;
		boolean out = this.layerStructure.addEdge(independent, dependent);

		this.preserveAcyclicity(independent, dependent);

		// sanity check
		assert this.variableInvariant();

		return out;
	}

	/**
	 * Indicates whether a dependency relation exists between the given layers
	 * 
	 * @param independent
	 *            Name of the layer it is desired to see if {@code dependent}
	 *            depends on
	 * @param dependent
	 *            Name of the layer it is desired to see depends on
	 *            {@code independent}
	 * @return {@code true if a relationship exists where {@code dependent}
	 *         depends on {@code independent}; {@code false} if the
	 *         {@code independent}&rarr;{@code dependent}
	 */
	public boolean containsDependency(String independent, String dependent) {
		return this.layerStructure.containsEdge(independent, dependent);
	}

	/**
	 * Gets the list of layers that depend on the specified layer
	 * 
	 * @param independent
	 *            name of the layer for which the dependent layers are desired
	 * @return A set containing the names of the layers that depend on
	 *         {@code independent}, or {@code null} if a layer with the given
	 *         name does not exist
	 */
	@SuppressWarnings("unchecked")
	public Set<String> getDependents(String independent) {
		return (Set<String>) (this.layerStructure.outGoing(independent));
	}

	/**
	 * Gets the list of layers that the specified layer depends on
	 * 
	 * @param dependent
	 *            name of the layer for which the independent layers are desired
	 * @return A set containing the names of the layers that {@code dependent}
	 *         depends on, or {@code null} if a layer with the given name does
	 *         not exist
	 */
	@SuppressWarnings("unchecked")
	public Set<String> getIndependents(String dependent) {
		return (Set<String>) (this.layerStructure.inComing(dependent));
	}

	/**
	 * Gets the dependency table for the relation {@code independent}&rarr;
	 * {@code dependent}
	 * 
	 * @param independent
	 *            Name of the layer {@code dependent} depends on in the desired
	 *            relation
	 * @param dependent
	 *            Name of the layer that depends on {@code independent} in the
	 *            desired relation
	 * @return A table with the dependence values for every variable in
	 *         {@code independent} (row) and {@code dependent} (column) or
	 *         {@code null} if the relation {@code independent}&rarr;
	 *         {@code dependent} does not exist
	 */
	public Double[][] getDependencyTable(String independent, String dependent) {
		return this.dependencyTables.get(new DirectedEdge(independent,
				dependent));
	}

	/**
	 * Removes a dependency relationship from the independent layer specified,
	 * to the dependent layer specified.
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} no longer depends
	 *            on
	 * @param dependent
	 *            the name of the layer that no longer depends on
	 *            {@code independent}
	 * @return the dependency table removed when the edge was removed, or
	 *         {@code null} if the relationship didn't exist
	 */
	public Double[][] removeDependency(String independent, String dependent) {
		boolean existed = this.layerStructure
				.removeEdge(independent, dependent);
		if (existed) {
			Double[][] out = this.dependencyTables.remove(new DirectedEdge(
					independent, dependent));

			// sanity check
			assert this.dependencyInvariant();
			assert this.variableInvariant();

			return out;
		} else {
			return null;
			// no change so no sanity check necessary
		}
	}

	/**
	 * Updates the list of variables associated to the given layer. Note that
	 * this will cause the dependency tables related to that layer to be
	 * reinitialized
	 * 
	 * @param layerName
	 *            the name of the layer to be replaced
	 * @param layerVariables
	 *            the new list of variables for this layer
	 * @return the previous list of variables associated with the specified
	 *         {@code layerName}
	 * @throws IllegalArgumentException
	 *             if no layer with that name exists
	 */
	@SuppressWarnings("unchecked")
	public List<String> replaceLayer(String layerName,
			List<String> layerVariables) throws IllegalArgumentException {
		layerMustExist(layerName);

		// reinitialize dependency tables
		Iterator<String> I = (Iterator<String>) (this.layerStructure
				.inComing(layerName).iterator());
		while (I.hasNext())
			this.initDependencyTable(I.next(), layerName);

		I = (Iterator<String>) (this.layerStructure.outGoing(layerName)
				.iterator());
		while (I.hasNext())
			this.initDependencyTable(layerName, I.next());

		// Update layer's actual value
		List<String> out = this.layerVariables.replace(layerName,
				layerVariables);

		// sanity check
		assert this.dependencyInvariant();
		assert this.variableInvariant();

		return out;

	}

	/**
	 * Updates the dependency table between the given layers to the given table.
	 * 
	 * @param independent
	 *            the name of the layer that {@code dependent} depends on
	 * @param dependent
	 *            the name of the layer that depends on {@code independent}
	 * @param dependencyTable
	 *            the new values for the dependency table of the relationship
	 *            between {@code independent} and {@code dependent}
	 * @return the dependency table with the previous values
	 * @throws IllegalArgumentException
	 *             if either one of the layers does not exist, if the dependency
	 *             table provided does not have enough entries for each
	 *             dimension, or if the rows of {@code dependencyTable} are of
	 *             different length
	 * @throws IllegalStateException
	 *             if the layers exist but the relationship between them has not
	 *             been created
	 */
	public Double[][] setDependency(String independent, String dependent,
			Double[][] dependencyTable) throws IllegalArgumentException,
			IllegalStateException {
		this.layerMustExist(independent);
		this.layerMustExist(dependent);
		if (!this.layerStructure.containsEdge(independent, dependent))
			throw new IllegalStateException(
					"A dependency relationship between "
							+ independent
							+ " and "
							+ dependent
							+ " does not exist in the specified DomainKnowledge instance. "
							+ "Use addDependency to add a new depdendency relationship.");
		this.rightSize(independent, dependent, dependencyTable);

		Double[][] out = this.dependencyTables.put(new DirectedEdge(
				independent, dependent), dependencyTable);

		// sanity check
		assert this.variableInvariant();

		return out;
	}

	/**
	 * Gets the relationships represented between all the variables of all the
	 * layers. Note that to perform this conversion, the same variable name
	 * can't appear in two different layers or indeed in the same layer
	 * 
	 * @return a graph where each vertex is a variable and each arc indicates if
	 *         there's a dependence between them greater than or equal to the
	 *         threshold. The graph is guaranteed to be acyclic.
	 * @throws IllegalStateException
	 *             if the variable names are not unique
	 * @throws NullPointerException
	 *             if any one of the {@link #dependencyTables}' cells have not
	 *             been initialized
	 * 
	 * @since 1.01 2016-03-14
	 */
	@SuppressWarnings("unchecked")
	public DirectedGraph variableDependency(Double threshold)
			throws IllegalStateException, NullPointerException {
		DirectedGraph out = new HashDirectedGraph(t(this.layerVariables.size()));

		Iterator<String> layerNames = this.layerVariables.keySet().iterator();
		while (layerNames.hasNext()) {
			String layer = layerNames.next();
			Set<String> dependsOn = (Set<String>) (this.layerStructure
					.inComing(layer));
			List<String> variables = this.layerVariables.get(layer);
			for (int j = 0; j < variables.size(); j++) {
				String v = variables.get(j);
				// create in the output graph
				boolean check = out.addVertex(v);
				if (!check)
					throw new IllegalStateException(
							"Duplicate detected! Variable "
									+ v
									+ " occurs in "
									+ layer
									+ " even though a variable with that name exists elsewhere.");

				// connect it with the other layers
				Iterator<String> D = dependsOn.iterator();
				while (D.hasNext()) {
					String parentLayer = D.next();
					Double[][] dependency = this.dependencyTables
							.get(new DirectedEdge(parentLayer, layer));
					List<String> parentVariables = this.layerVariables
							.get(layer);
					for (int i = 0; i < parentVariables.size(); i++) {
						if (dependency[i][j].compareTo(threshold) >= 0) {
							check = out.addEdge(parentVariables.get(i), v);
							assert check;
						}
					}// end for of parentVariables
				}// end while of parentLayers
			}// end for of variables
		}// end while of layerNames

		return out;
	}

	/**
	 * Gets the relationships represented between all the variables of all the
	 * layers assuming the default threshold of zero (any positive number is
	 * sufficient). Note that to perform this conversion, the same variable name
	 * can't appear in two different layers or indeed in the same layer
	 * 
	 * @return a graph where each vertex is a variable and each arc indicates if
	 *         there's a positive dependence between them. The graph is
	 *         guaranteed to be acyclic
	 * @throws IllegalStateException
	 *             if the variable names are not unique
	 * @throws NullPointerException
	 *             if any one of the {@link #dependencyTables}' cells have not
	 *             been initialized
	 * @since 1.01 2016-03-14
	 */
	public DirectedGraph variableDependency() throws IllegalStateException,
			NullPointerException {
		return this.variableDependency(new Double(0));
	}

	/**
	 * Tests the class' correctness for the given number of layers. Each layer
	 * is created with as many variables as the layer name (thus, layer 0 has
	 * zero variables, layer 1 has one, etc).
	 * 
	 * @throws AssertionError
	 *             if any of the checks fail
	 */
	private void tddTest(int layers) throws AssertionError {
		boolean flagRaised;
		System.err.println("Validating class...");
		// adding and removing layers
		for (int i = 0; i < layers; i++)
			this.addLayer(Integer.toString(i), null);
		for (int i = 0; i < layers; i++) {
			flagRaised = false;
			try {
				this.addLayer(Integer.toString(i), null);
			} catch (IllegalArgumentException e) {
				flagRaised = true;
			}
			assert flagRaised;
		}
		for (int i = 0; i < layers; i++)
			this.removeLayer(Integer.toString(i));
		assert this.layerVariables.size() == 0;
		assert this.layerStructure.vertices().size() == 0;
		System.err.println("Passed layer test");

		// create "real" layers (each layer having as many elements as its name;
		// this ensures the edge case is tested)
		for (int i = 0; i < layers; i++) {
			List<String> V = new Vector<String>(i);
			for (int j = 0; j < i; j++)
				V.add(Integer.toString(j));
			this.addLayer(Integer.toString(i), V);
		}
		this.addLayer(Integer.toString(layers), null); // other edge case to
														// test

		// Cycle detection
		for (int i = 1; i <= layers; i++) {
			this.addDependency(Integer.toString(i - 1), Integer.toString(i));
			flagRaised = false;
			try {
				this.addDependency(Integer.toString(i - 1), Integer.toString(0));
			} catch (SecurityException e) {
				flagRaised = true;
			}
			assert flagRaised;
		}
		// no multigraphs
		for (int i = 1; i <= layers; i++) {
			flagRaised = false;
			try {
				this.addDependency(Integer.toString(i - 1), Integer.toString(i));
			} catch (IllegalStateException e) {
				flagRaised = true;
			}
			assert flagRaised;
		}
		// vertex must exist
		flagRaised = false;
		try {
			this.addDependency(Integer.toString(-1), Integer.toString(0));
		} catch (IllegalArgumentException e) {
			flagRaised = true;
		}
		assert flagRaised;
		flagRaised = false;
		try {
			this.addDependency(Integer.toString(layers),
					Integer.toString(layers + 1));
		} catch (IllegalArgumentException e) {
			flagRaised = true;
		}
		assert flagRaised;

		// dependency removal
		for (int i = 1; i <= layers; i++) {
			this.removeDependency(Integer.toString(i - 1), Integer.toString(i));
			assert this.removeDependency(Integer.toString(i - 1),
					Integer.toString(0)) == null;
		}
		assert this.layerStructure.numEdges() == 0;
		System.err.println("Passed dependences test");

		// creation of "real" relations
		for (int i = 0; i < layers; i++) {
			for (int j = i + 1; j <= layers; j++) {
				this.addDependency(Integer.toString(i), Integer.toString(j));
			}
		}
		// dependency table replacement
		for (int i = 0; i < layers; i++) {
			for (int j = i + 1; j <= layers; j++) {
				if (j < layers)
					this.setDependency(Integer.toString(i),
							Integer.toString(j), new Double[i][j]);
				else
					this.setDependency(Integer.toString(i),
							Integer.toString(j), new Double[i][0]);

				if (i != j) {
					flagRaised = false;
					try {
						this.setDependency(Integer.toString(i),
								Integer.toString(j), new Double[j][i]);
					} catch (IllegalArgumentException e) {
						flagRaised = true;
					}
					assert flagRaised;
				}

				// for links that don't exist
				flagRaised = false;
				try {
					this.setDependency(Integer.toString(j),
							Integer.toString(i), new Double[j][i]);
				} catch (IllegalStateException e) {
					flagRaised = true;
				}
				assert flagRaised;
			}
		}
		System.err.println("Passed dependency tables test");

		// TODO: Layer replacement
		// System.err.println("Passed blanking test");

		System.err.println("Validation complete");
	}

	/**
	 * Performs a full correctness test to ensure the class is working
	 * correctly, then presents
	 * 
	 * @param args
	 * @throws Exception
	 *             if the correctness test fails
	 */
	// TODO: parse command line arguments
	public static void main(String[] args) throws Exception {
		DomainKnowledge d = new DomainKnowledge();
		d.tddTest(10);
		d = new DomainKnowledge();

		// TODO: Interactive console
	}

}
