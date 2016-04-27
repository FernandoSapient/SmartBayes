/**Contains classes to build economic models using World Bank data
 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version	0.08 2016-04-25
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.08</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link ReconstructionTest#main(String[])} now sends summary
 * 			statistics to a specified CSV
 * 			({@link ReconstructionTest} is now version 0.4)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.07</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link ReconstructionTest#main(String[])} now skips countries
 * 			with insufficient data to discretize (Previously, a compatibility
 * 			error was thrown) 
 * 			({@link ReconstructionTest} is now version 0.3)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.06</td>
 * 		<td>2016-04-25</td>
 * 		<td>Bugfixes; {@link ModelClusterizer#shiftBy(java.util.List, int)} now
 * 			uses {@code NaN} instead of {@code null} as the filter element
 * 			for compatibility with Weka.
 * 			({@link ModelClusterizer} is now version 0.5;
 * 			{@link ReconstructionTest} is now version 0.2)</td>
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