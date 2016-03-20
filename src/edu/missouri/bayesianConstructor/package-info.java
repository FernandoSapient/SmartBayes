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
 * @version	0.02 2016-03-14
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.03</td>
 * 		<td>2016-03-18</td>
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