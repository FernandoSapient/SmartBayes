//TODO: add support for converting to il2 bayesian network and Weka bayesian network
//		(may require changing Graph library)
/**Contains classes to automatically build a Bayesian Network.
 * Typical usage is to create a model of the {@link DomainKnowledge},
 * export this into a {@link edu.ucla.belief.BeliefNetwork} (from
 * <a href="http://reasoning.cs.ucla.edu/samiam">SamIam</a>), and then
 * train or evaluate the belief network representation using methods
 * from another package. The programmer is  encouraged to try 
 * exporting the Domain Knowledge model into other structures, 
 * but no support for them is provided in this package at the present time.
 *  
 * 
 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version	0.19 2016-04-19
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.19</td>
 * 		<td>2016-04-19</td>
 * 		<td>Added {@link Main#loadCSVwithFiltering(java.io.Reader, String, int)},
 * 			{@link Main#loadCSVwithFiltering(java.io.Reader, String)}, and
 * 			{@link Main#loadCSVwithFiltering(java.io.Reader)}
 * 			to permit other callers to use it.
 * 			({@link Main} is now version 1.12)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.18</td>
 * 		<td>2016-04-18</td>
 * 		<td>Added {@link Main#networkToFile(edu.ucla.belief.BeliefNetwork, String)}
 * 			to permit other callers to use it, added compatibility for regular 
 * 			expressions and filtering of any column.
 * 			({@link Main} is now version 1.11)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.17</td>
 * 		<td>2016-04-18</td>
 * 		<td>Improved documentation in {@link DomainKnowledge} and {@link Main};
 * 			added {@link Main#graphToNetwork(edu.ucla.structure.DirectedGraph, String[])
 * 			as a true successor function to
 * 			{@link Main#graphToNodeGraph(edu.ucla.structure.DirectedGraph, String[])}.
 * 			({@link DomainKnowledge} is now version 1.05
 * 			({@link Main} is now version 1.10)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.16</td>
 * 		<td>2016-04-09</td>
 * 		<td>Added {@link DomainKnowledge#layerMap()} to allow callers to use
 * 			both {@link Main#graphToNetwork} methods securely
 * 			({@link DomainKnowledge} is now version 1.04)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.15</td>
 * 		<td>2016-04-08</td>
 * 		<td>Added {@link Main#graphToNetwork(edu.ucla.structure.DirectedGraph, 
 * 							String[], java.util.Map, char, double)}
 * 			to allow the caller to specify the parameters of the {@link NodePlacer}
 * 			to use. 
 * 			({@link Main} is now version 0.09)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.14</td>
 * 		<td>2016-03-30</td>
 * 		<td>Fixed {@link NodePlacer#STAR} mode.
 * 			{@link NodePlacer} is now version 1.4.</td>
 * 	</tr>
 * 		<td>0.13</td>
 * 		<td>2016-03-29</td>
 * 		<td>Fixed {@link NodePlacer#GEOMETRIC} mode.
 * 			{@link NodePlacer} is now version 1.3.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.12</td>
 * 		<td>2016-03-29</td>
 * 		<td>Fixed {@link NodePlacer#LAYERED} mode.
 * 			{@link NodePlacer} is now version 1.2.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.11</td>
 * 		<td>2016-03-29</td>
 * 		<td>Added {@link NodePlacer#ASTERISK} mode after a happy accident.
 * 			{@link NodePlacer} is now version 1.1.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.10</td>
 * 		<td>2016-03-29</td>
 * 		<td>Added {@link NodePlacer} to help ouputted files be more readable in SamIam viewer.
 * 			{@link Main} is now version 0.08.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.09</td>
 * 		<td>2016-03-21</td>
 * 		<td>{@link Main#main(String[])} now receives output file as a parameter
 * 			({@link Main} is now version 0.07)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.08</td>
 * 		<td>2016-03-21</td>
 * 		<td>Fixed {@link Main#graphToNodeGraph(edu.ucla.structure.DirectedGraph,
 * 			String[])} to account for hashing order and {@code BeliefNetwork} 
 * 			limitations ({@link Main} is now version 0.06)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.07</td>
 * 		<td>2016-03-20</td>
 * 		<td>Added {@link Main#graphToNodeGraph(edu.ucla.structure.DirectedGraph,
 *			String[])} to follow SamIam conventions ({@link Main} is now
 * 			version 0.05)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.06</td>
 * 		<td>2016-03-19</td>
 * 		<td>Fixed {@link Main#dependency(java.util.List, java.util.List)} which
 * 			was not ignoring {@code null} values correctly ({@link Main} is now
 * 			version 0.04)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.05</td>
 * 		<td>2016-03-18</td>
 * 		<td>Added added CSV support to {@link Main#Main()} through
 * 			<a href="http://opencsv.sourceforge.net">openCSV</a> (now version 0.03)</td>
 * 	</tr>
 * 		<td>0.04</td>
 * 		<td>2016-03-17</td>
 * 		<td>Fixed {@link Main#dependency(java.util.List, java.util.List)} which
 * 			was not ignoring {@code null} values correctly; and brought {@link
 * 			DomainKnowledge} up to version 1.03, improving handling of
 * 			uninitialized dependency tables.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.03</td>
 * 		<td>2016-03-15</td>
 * 		<td>added {@link Main} (version 0.01) and debugged {@link
 * 			DomainKnowledge#variableDependency(Double)} ({@link DomainKnowledge}
 * 			is now version 1.02)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.02</td>
 * 		<td>2016-03-14</td>
 * 		<td>added {@link DomainKnowledge#variableDependency(Double)} ({@link
 * 			DomainKnowledge} is now version 1.01)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.01</td>
 * 		<td>2016-03-13</td>
 * 		<td>{@link DomainKnowledge} version 1.0 finalized</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.00</td>
 * 		<td>2016-02-03</td>
 * 		<td>Package created</td>
 * 	</tr>
 * </table>
 */
package edu.missouri.bayesianConstructor;