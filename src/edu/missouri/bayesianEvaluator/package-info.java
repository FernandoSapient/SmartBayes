/**Contains classes to train and evaluate a Bayesian Network.
 * Bayesian networks may be manually constructed (using software
 * such as <a href=http://www.cs.cmu.edu/~javabayes/>JavaBayes</url>
 * or <a href="http://reasoning.cs.ucla.edu/samiam">SamIam</a>)
 * or algorithmically (using methods like
 * <a href=../bayesianConstructor/DomainKnowledge#variableDependency(Double)>{@code 
 * bayesianConstructor.DomainKnowledge.variableDependency(Double)}</a>). The network
 * need not be trained (i.e. the conditional probability tables don't
 * need to have been computed) 
 * 
 * @author	<a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a> 
 * @version	0.03 2016-04-03
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
 * 		<td>2016-04-02</td>
 * 		<td>Modified {@link Trainer#main(String[])} in an attempt to remove nodes
 * 			that exist in the data file, but not in the Bayesian network.
 * 			{@link Trainer} is now version 0.02</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.02</td>
 * 		<td>2016-04-02</td>
 * 		<td>Added {@link Trainer}; moved XMLBIF loading calls in
 * 			{@link BifUpdate#main(String[])} to {@link BifUpdate#loadBayesNet(String)}.
 * 			{@link BifUpdate} is now version 0.02</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.01</td>
 * 		<td>2016-04-01</td>
 * 		<td>Package created with {@link BifUpdate}</td>
 * 	</tr>
 * </table>
 */
package edu.missouri.bayesianEvaluator;