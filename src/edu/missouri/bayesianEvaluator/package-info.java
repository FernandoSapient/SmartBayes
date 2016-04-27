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
 * @version	0.23 2016-04-25
 * 
 * <h2>Version history</h2>
 * <table>
 * 	<tr>
 * 		<th>Ver#</th>
 * 		<th>date</th>
 * 		<th>Changes</th>
 * 	</tr>
 * 	<tr>
 * 		<td>0.23</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link Trainer#discretizeToBayes(weka.core.Instances, weka.classifiers.bayes.BayesNet, boolean)}
 * 			{@code ArithmeticException} removed; until better handling can be implemented,
 * 			insufficient data will now be handled as no data
 * 			({@link Trainer} is now version 0.14)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.22</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link Trainer#discretizeToBayes(weka.core.Instances, weka.classifiers.bayes.BayesNet, boolean)}
 * 			now throws an {@code ArithmeticException} if there's not enough data to discretize
 * 			(previously an assertion error was thrown)
 * 			({@link Trainer} is now version 0.13)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.21</td>
 * 		<td>2016-04-25</td>
 * 		<td>{@link Trainer#discretizeToBayes(weka.core.Instances, weka.classifiers.bayes.BayesNet, boolean)}
 * 			now accounts for attributes with all values missing (usually due to filtering)
 * 			({@link Trainer} is now version 0.12)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.20</td>
 * 		<td>2016-04-23</td>
 * 		<td>Added {@link Trainer#addAttributeAt(weka.core.Instances, weka.core.Attribute, int, double[])}
 * 			({@link Trainer} is now version 0.11)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.19</td>
 * 		<td>2016-04-23</td>
 * 		<td>Added {@link Evaluator#allAttributesAccuracies(edu.ucla.belief.BeliefNetwork, weka.core.Instances, java.util.Map)}
 * 			({@link Evaluator} is now version 0.10)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.18</td>
 * 		<td>2016-04-23</td>
 * 		<td>Added {@link Evaluator#crossValidationAccuracies(weka.classifiers.bayes.net.EditableBayesNet, java.util.Map, String)}
 * 			({@link Evaluator} is now version 0.09)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.17</td>
 * 		<td>2016-04-23</td>
 * 		<td>{@link Evaluator#main}
 * 			now prints summarized statistics, rather than one for each split
 * 			({@link Evaluator} is now version 0.08)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.16</td>
 * 		<td>2016-04-22</td>
 * 		<td>{@link Evaluator#accuracy(edu.ucla.belief.BeliefNetwork, weka.core.Instances)}
 * 			now gives an error when dividing by zero (formerly returned {@code Infinity})
 * 			({@link Evaluator} is now version 0.07)</td>
 * 	</tr>
 * 	<tr>
 * 		<td>0.15</td>
 * 		<td>2016-04-22</td>
 * 		<td>added {@link Evaluator#accuracy(edu.ucla.belief.BeliefNetwork, weka.core.Instances)}
 * 			({@link Evaluator} is now version 0.06)</td>
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