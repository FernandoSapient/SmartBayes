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
 * @version	0.14 2016-04-21
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.14</td>
 * 		<td>2016-04-22</td>
 * 		<td>Bugfixes
 * 			({@link Evaluator} is now version 0.05)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.13</td>
 * 		<td>2016-04-21</td>
 * 		<td>added {@link Evaluator#loadEvidenceFromWeka(edu.ucla.belief.BeliefNetwork, weka.core.Instance)},
 * 			{@link Evaluator#loadSamiamBayes(String)}, and
 * 			{@link Evaluator#shenoyShaferMarginals(edu.ucla.belief.BeliefNetwork, weka.core.Instance)}
 * 			({@link Evaluator} is now version 0.04)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.12</td>
 * 		<td>2016-04-20</td>
 * 		<td>added {@link Trainer#trainToFile} and {@link Evaluator#wekaEvaluation}
 * 			({@link Trainer} is now version 0.10; {@link Evaluator} is now version 0.03)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.11</td>
 * 		<td>2016-04-19</td>
 * 		<td>Security enhancement: {@link BifUpdate#loadBayesNet(String)} now returns an
 * 			<em>editable</em> Bayes net to encourage the use of {@code setData}.
 * 			{@link BifUpdate} is now version 0.03; {@link Trainer} is now version 0.09;
 * 			{@link Evaluator} is now version 0.02.</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.10</td>
 * 		<td>2016-04-19</td>
 * 		<td>Added {@link Evaluator}</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.09</td>
 * 		<td>2016-04-10</td>
 * 		<td>Added {@link Trainer#conformToNetwork(weka.core.Instances, weka.classifiers.bayes.BayesNet, boolean)}.
 * 			{@link Trainer} is now version 0.08</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.08</td>
 * 		<td>2016-04-09</td>
 * 		<td>Added {@link Trainer#reorderAttributes(weka.core.Instances, java.util.List)}.
 * 			to reduce the chance that {@code weka.classifiers.bayes.BayesNet.estimateCPTs()}
 * 			will swap the node names.
 * 			{@link Trainer} is now version 0.07</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.07</td>
 * 		<td>2016-04-08</td>
 * 		<td>Improved user feedback in {@link Trainer#main(String[])}.
 * 			{@link Trainer} is now version 0.06</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.06</td>
 * 		<td>2016-04-08</td>
 * 		<td>Added
 * 				{@link Trainer#filterByCriterion(String, weka.core.Instances, int)}
 * 				{@link Trainer#restrictToAttributeSet(weka.core.Instances, java.util.Collection)}
 * 				{@link Trainer#discretizeToBayes(weka.core.Instances, weka.classifiers.bayes.BayesNet, boolean)}
 * 				and {@link Trainer#getAttributeNames(weka.core.Instances)}
 * 			to allow callers to do what {@link Trainer#main(String[])} does.
 * 			{@link Trainer} is now version 0.05</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.05</td>
 * 		<td>2016-04-06</td>
 * 		<td>{@link Trainer#main(String[])} now outputs results.
 * 			{@link Trainer} is now version 0.04</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.04</td>
 * 		<td>2016-04-02</td>
 * 		<td>{@link Trainer#main(String[])} now runs with no conflicts.
 * 			{@link Trainer} is now version 0.03</td>
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