/**Contains classes to automatically build and evaluate a Bayesian Network.
 * Typical usage is to create a model of the {@link DomainKnowledge},
 * export this into a {@link edu.ucla.belief.BeliefNetwork} or
 * {@link il2.model.BayesianNetwork} (from
 * <a href="http://reasoning.cs.ucla.edu/samiam">SamIam</a>), and then
 * train or evaluate the belief network representation. The reaader is
 * encouraged to try other exportations, but no support for them is
 * provided in this package.
 *  
 * 
 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version	0.07 2016-03-20
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.07</td>
 * 		<td>2016-03-20</td>
 * 		<td>Added {@link Main#graphToNodeGraph(edu.ucla.structure.DirectedGraph, String[])} to follow SamIam conventions</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.06</td>
 * 		<td>2016-03-19</td>
 * 		<td>Fixed {@link Main#dependency(java.util.List, java.util.List)} which was not ignoring {@code null} values correctly</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.05</td>
 * 		<td>2016-03-18</td>
 * 		<td>Added sample CSV data and openCSV library, added CSV support to {@link Main#Main()}</td>
 * 	</tr>
 * 		<td>0.04</td>
 * 		<td>2016-03-17</td>
 * 		<td>Fixed {@link Main#dependency(java.util.List, java.util.List)} which was not ignoring {@code null} values correctly</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.03</td>
 * 		<td>2016-03-15</td>
 * 		<td>added {@link Main}</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.02</td>
 * 		<td>2016-03-14</td>
 * 		<td>added {@link DomainKnowledge#variableDependency(Double)}</td>
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