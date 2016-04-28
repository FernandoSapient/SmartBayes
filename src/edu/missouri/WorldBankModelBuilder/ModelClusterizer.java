package edu.missouri.WorldBankModelBuilder;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import com.opencsv.CSVReader;

import edu.missouri.bayesianConstructor.DomainKnowledge;
import edu.missouri.bayesianConstructor.NodePlacer;
import edu.missouri.bayesianConstructor.Main;
import edu.ucla.belief.BeliefNetwork;
import edu.ucla.structure.DirectedGraph;

/**
 * Provides a way to generate all the economic models, and group countries by it
 *
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 0.06 2016-04-27
 * @since {@code WorldBankModelBuilder} version 0.01 2016-04-19
 */
public class ModelClusterizer {

	/**
	 * Gets the set of all countries in the given file
	 * 
	 * @param filename
	 *            Name of the file to be read
	 * @param column
	 *            Index of the column containing the countries
	 * @return
	 * @throws ArrayIndexOutOfBoundsException
	 *             if the file does not have enough columns for {@code column}
	 *             to be a valid index
	 * @throws IOException
	 *             if at any point it cannot read the next line of the file
	 * @since 0.01 2016-04-19
	 */
	public static Set<String> getCountries(String filename, int column)
			throws ArrayIndexOutOfBoundsException, IOException {
		CSVReader reader = new CSVReader(new FileReader(filename));

		Set<String> out = new HashSet<String>();

		String[] nextLine;
		reader.readNext(); // discard header row
		while ((nextLine = reader.readNext()) != null) {
			String country = nextLine[column];
			if (country != null && country.length() > 0)
				out.add(country);
		}

		reader.close();
		return out;
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
	 *            An array of length 4 or 5, where the first position contains
	 *            the name of the file containing the input data, the second
	 *            contains a directory to save the resulting networks in, the
	 *            third position indicates whether to use the Unesco model
	 *            ({@code true}) or the Smets-Woulters model ({@code false}), the
	 *            fourth position optionally contains the index with the values
	 *            to group the results by (zero by default), and the fifth
	 *            position (in the case of length 5) contains one of the
	 *            {@link NodePlacer} configuration codes.
	 * @throws IOException
	 *             if the input file could not be read
	 * @throws FileNotFoundException
	 *             if the output file could not be created
	 * @throws NumberFormatException
	 *             if {@code args[2]} is not a valid index because it is not a
	 *             number
	 * @throws ArrayIndexOutOfBoundsException
	 *             if {@code args[2]} is not a valid index because it does not
	 *             fall within the dataset
	 */
	// improve parameter handling. See JCLAP for potential solution
	public static void main(String[] args) throws IOException,
			FileNotFoundException, NumberFormatException, ArrayIndexOutOfBoundsException {
		if (args.length < 2) {
			System.err
					.println("Usage: java Main <input data file> <output directory> <Use Unesco Model> <group-by column> [plot mode]");
			return;
		}
		String filename = args[0];
		boolean useUnesco = Boolean.parseBoolean(args[2]);
		int groupByIndex = Integer.parseInt(args[3]);

		Set<String> countries = getCountries(filename, groupByIndex);
		Iterator<String> C = countries.iterator();
		Map<DirectedGraph, List<String>> clustering = new HashMap<DirectedGraph, List<String>>();

		// TODO: perform in a method to allow others to call it and get their
		// clusterings
		while (C.hasNext()) {
			String country = C.next().trim();
			Map<String, List<Double>> data = Main.loadCSVwithFiltering(
					new FileReader(filename), country, groupByIndex);

			DomainKnowledge m;
			if(useUnesco)
				m = buildUnescoModel(data);
			else
				m = buildSWModel(data);

			DirectedGraph variableGraph = m.variableDependency(.03);
			if (!clustering.containsKey(variableGraph))
				clustering.put(variableGraph, new Vector<String>());
			clustering.get(variableGraph).add(country);

			// define values
			String[] values = { "low", "med", "high" };

			// convert to bayesian network
			char config = args.length > 4 ? args[4].charAt(0)
					: NodePlacer.LAYERED;
			Map<String, List<String>> orderedLayers = new LinkedHashMap<String, List<String>>(
					m.layerSet().size());
			if (args.length > 4 && config == NodePlacer.STAR) {
				// "satellite dish" display
				orderedLayers.put("spacer1", new Vector<String>());
				orderedLayers.put("spacer2", new Vector<String>());
				orderedLayers.put("Economic", m.getLayer("Economic"));
				orderedLayers.put("spacer3", new Vector<String>());
				orderedLayers.put("spacer4", new Vector<String>());
			}
			if(useUnesco){
				orderedLayers.put("Previous Economy",
						m.getLayer("Previous Economy"));
				orderedLayers.put("Education", m.getLayer("Education"));
				orderedLayers.put("Innovation", m.getLayer("Innovation"));
				orderedLayers.put("Production", m.getLayer("Production"));
			}else{
				orderedLayers.put("PrevResource", m.getLayer("PrevResource"));
				orderedLayers.put("PrevEstimation", m.getLayer("PrevEstimation"));
				orderedLayers.put("Resource", m.getLayer("Resource"));
				orderedLayers.put("Estimation", m.getLayer("Estimation"));
			}
			if (args.length <= 4 || config != NodePlacer.STAR) {
				orderedLayers.put("Economic", m.getLayer("Economic"));
			}

			BeliefNetwork out;
			if (args.length > 4)
				out = Main.graphToNetwork(variableGraph, values, orderedLayers,
						config, 0);
			else
				out = Main.graphToNetwork(variableGraph, values, orderedLayers);

			String fileOut = args[1] + "/" + country + ".xml";
			// This is bad practice
			// TODO: use input parameter to determine target directory
			boolean result = Main.networkToFile(out, fileOut);
			if (result)
				System.out.println("File \"" + fileOut
						+ "\" created successfully");
			else
				System.err.println("Could not write file \"" + fileOut + "\".");
			// but does it change with the years?
		}

		System.out
				.println(clustering.size() + " distinct networks were built:");
		Iterator<DirectedGraph> I = clustering.keySet().iterator();
		while (I.hasNext())
			System.out.println("Cluster: " + clustering.get(I.next()));
		// TODO: Print only those clusters that have more than one element
		// TODO: can we cluster hierarchically? (score by average inter-layer
		// degree?)
		// TODO: check if this is more efficient if we iterate over the
		// entrysets

	}

	/**
	 * Builds a full {@link DomainKnowledge} model following the structure
	 * proposed in the UNESCO world engineering report. This method is included
	 * as sample code and will be deprecated as soon as file-reading support is
	 * added
	 * <p/>
	 * Note: this method is not a clone of {@link edu
	 * /missouri/BayesianConstructor#buildUnescoModel(Map)}; this method adds a
	 * variable: previous year
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

		// Create previous year category
		// TODO: offer this as an option for when it's read from a file
		data.put("Previous " + PPP, shiftBy(data.get(PPP), 1));
		data.put("Previous " + growth, shiftBy(data.get(growth), 1));

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
				Arrays.asList("Previous " + growth, "Previous " + PPP));
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
	 * @since 0.06 2016-04-27
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
		String prevConsump = "Previous " + consump;
		String prevInvest = "Previous " + invest;
		String prevCapital = "Previous " + capital;
		String prevWages = "Previous " + wages;
		String prevForm = "Previous " + form;

		// Create previous year category
		// TODO: offer this as an option for when it's read from a file
		data.put(prevConsump, shiftBy(data.get(consump), 1));
		data.put(prevInvest, shiftBy(data.get(invest), 1));
		data.put(prevCapital, shiftBy(data.get(capital), 1));
		data.put(prevWages, shiftBy(data.get(wages), 1));
		data.put(prevForm, shiftBy(data.get(form), 1));

		// Hardwire 3-layer structure
		// TODO: read domain knowledge structure from file
		DomainKnowledge m = new DomainKnowledge();
		m.addLayer("Resource", Arrays.asList(wages, interest, form, ratio));
		m.addLayer("Estimation", Arrays.asList(consump, invest, worker, capital, exog));
		m.addLayer("Economic", Arrays.asList(GDP));
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
		m.addDependency("Estimation", "Economic",
				Main.getDependency(estimation, economy));
		return m;
	}

	/**
	 * Creates a copy of the list where all element indices are shifted ahead by
	 * the amount indicated. The size of the list is preserved: Elements near
	 * the end are dropped off, and {@code NaN} is used as a filler element
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
	 * @since 0.01 2016-04-19
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
}
