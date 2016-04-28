/**Contains classes to build economic models using World Bank data
 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version	0.10 2016-04-27
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.10</td>
 * 		<td>2016-04-28</td>
 * 		<td>Moved {@link ModelClusterizer#shiftBy(java.util.List, int)}
 * 			and {@link ReconstructionTest#addPrevious(weka.core.Instances, String)}
 * 			to {@link edu/missouri/Trainer}
 * 			({@link ModelClusterizer} is now version 0.07;
 * 			{@link Evaluator} is now version 0.06)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.09</td>
 * 		<td>2016-04-27</td>
 * 		<td>Added support for the Smets-Woulters economic model as a
 * 			temporary fix to not being able to read domain knowledge
 * 			from files:
 * 			<ul>
 * 				<li>added {@link ModelClusterizer#buildSWModel(java.util.Map)}</li>
 * 				<li>added %lt;use Unesco model&gt; parameter to {@link ModelClusterizer#main(String[])}</li>
 * 				<li>added {@link ReconstructionTest#getSWTitles()}
 * 				<li>added {@link ReconstructionTest#buildSWModel(java.util.Map)}
 * 				<li>added {@link ReconstructionTest#buildEmptySWModel(java.util.Map)}
 * 				<li>added {@link ReconstructionTest#constructSWToFile(weka.core.Instances, String, int)}
 * 				<li>added {@link ReconstructionTest#constructEmptySWToFile(weka.core.Instances, String, int)}
 * 				<li>added %lt;use Unesco model&gt; parameter to {@link ModelClusterizer#main(String[])}</li>
 * 			</ul>
 * 			({@link ModelClusterizer} is now version 0.06 {@link ReconstructionTest} is now version 0.05)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.08</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link ReconstructionTest#main(String[])} now sends summary
 * 			statistics to a specified CSV
 * 			({@link ReconstructionTest} is now version 0.04)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.07</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link ReconstructionTest#main(String[])} now skips countries
 * 			with insufficient data to discretize (Previously, a compatibility
 * 			error was thrown) 
 * 			({@link ReconstructionTest} is now version 0.03)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.06</td>
 * 		<td>2016-04-25</td>
 * 		<td>Bugfixes; {@link Trainer#shiftBy(java.util.List, int)} now
 * 			uses {@code NaN} instead of {@code null} as the filter element
 * 			for compatibility with Weka.
 * 			({@link ModelClusterizer} is now version 0.05;
 * 			{@link ReconstructionTest} is now version 0.02)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.05</td>
 * 		<td>2016-04-24</td>
 * 		<td>Added {@link ReconstructionTest}</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.04</td>
 * 		<td>2016-04-19</td>
 * 		<td>Added {@link ModelClusterizer#buildUnescoModel(java.util.Map)}
 * 			{@link ModelClusterizer} now allows specifying the output directory</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.03</td>
 * 		<td>2016-04-19</td>
 * 		<td>{@link ModelClusterizer} now stores in ./out/nets</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.02</td>
 * 		<td>2016-04-19</td>
 * 		<td>Bug fixes</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.01</td>
 * 		<td>2016-04-19</td>
 * 		<td>Package created with {@link ModelClusterizer}</td>
 * 	</tr>
 * </table>
 */
package edu.missouri.WorldBankModelBuilder;
import edu.missouri.bayesianEvaluator.Trainer;
