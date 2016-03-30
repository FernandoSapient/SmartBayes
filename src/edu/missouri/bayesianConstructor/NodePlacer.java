/**
 * 
 */
package edu.missouri.bayesianConstructor;

import java.awt.Point;

/**
 * The {@code NodePlacer} class computes the best placement for the given node
 * on the canvas on which its graph is displayed. The class works primarily
 * under the assumption that the nodes can be grouped into 
 * <strong>components</strong> such as the "layers" in a {@link DomainKnowledge}  
 * model. Components must be assigned a fixed order before calling, giving each
 * an immediate predecessor and an immediate successor in the order (which need 
 * not be related to how the components connect).
 * <p/>
 * The class can plot several configurations:
 * <ul>
 * 		<li><b>Geometric:</b> the class can place the nodes following a geometric
 * 			shape. In this scenario, each component is assigned one of the sides, and
 * 			there are as many sides as they are components (for example, if nodes are
 * 			grouped into three components, they will be placed along the border of a
 * 			triangle). This configuration is preferred when there may be relations
 * 			among the members of all the components.
 * 		</li>
 * 		<li><b>Layered:</b> the class can place the nodes stacked by component.
 * 			Under this configuration, all the  nodes of a component are placed along
 * 			a line, and every other component's nodes follow a line parallel to it.
 * 			This configuration is preferred when relations are restricted to
 * 			the immediately preceding and the immediately following component. 
 * 		</li>
 * 		<li><b>Asterisk:</b> the class places each component along a line radiating
 * 			from the center. 
 * 			This configuration is preferred when relations are restricted to
 * 			the immediately preceding and the immediately following component,
 * 			with possible relations between the first and last components. 
 * 		</li>
 * 		<li><b>Star:</b> The members of a component are placed along a circle
 * 			The circles are also placed along a circle, except for the first component
 * 			which is placed in the center. This configuration is preferred if there
 * 			is one central component which is known to be the source of all others,
 * 			or which is known to be the sink of all others, and every other component
 * 			interacts with this central component; and optionally, with the one 
 * 			immediately preceding it, and the one immediately following it. It is 
 * 			<em>not</em> recommended if, for example, there is a five-component 
 * 			structure and the second component interacts with the third, as this will 
 * 			cause arcs to cut through the center component.
 * 		</li>
 * </ul>
 * The placements are affected by an {@link #angle} and a {@link leeway} parameter.
 * The angle affects the initial orientation of the graph; for instance, whether a
 * star cluster should have a point directly above the center, or whether the
 * lines described by the components are horizontal, vertical, or diagonal. The leeway
 * parameter establishes how much space should be given to each node. By default,
 * leeway is set to 128 pixels, which is a good minimum size for the
 * <a href="http://reasoning.cs.ucla.edu/samiam">SamIam</a> viewer. For other
 * viewers (such as the Weka viewer which does not truncate the variable name)
 * larger sizes may be desired.
 * <p/>
 * The {@link #angle} parameter can also be used to specify direction in the 
 * case of layered placement: an angle of &pi;/2 will place each component's nodes
 * on a vertical line, with its successor to its right, while an angle of -&pi;/2
 * (or 3&pi;/2) will place each component's nodes on a vertical line, with its
 * successor to its left.
 * <p/>
 * The {@link #canvas} size is an important parameter to determine the positions
 * as it will indicate the center of the plot. Because this is often hard to
 * estimate, constructors are provided that perform a calculation of the estimated
 * space needed, such as {@link #NodePlacer(int, int, int, double, char)}.
 * Note that if canvas size is not properly set, negative values may be
 * returned as the class tries to enforce the leeway.
 * <p/>
 * Once initialized, the class will give the computed position for each node
 * by calling {link #position(int, int, int, int)}. Using a larger number of components than
 * what will actually be graphed can allow creativity in the result. For example, 
 * a graph may be made to look like a satellite dish by using the "star" 
 * configuration and specifying twice the number of components it actually has.
 * <p/>
 * Parameters are defined as {@code final} to prevent callers from changing them
 * and causing erratic behavior.
 * 
 * @author <a href="mailto:fthc8@missouri.edu">Fernando J. Torre-Mora</a>
 * @version 1.4 2016-03-30
 * @since {@link bayesianConstructor} version 0.10 2016-03-29
 */
public class NodePlacer {
	/**Specifies the angle (in radians) at which all placement computations
	 * should start. An angle of zero specifies a horizontal line, while an
	 * angle of {@link Math#PI}/2 specifies a vertical line going up.
	 * The default angle is zero radians.
	 * Note that once this value is set, it can no
	 * longer be changed.
	 */
	public final double angle;
	private static final double DEFAULT_ANGLE = 0.0;
	
	/**Number of pixels to leave unoccupied around each node.
	 * Should be greater than the size a node is expected to take up
	 * in the final placing. The default leeway is 128.
	 * Note that once this value is set, it can no
	 * longer be changed.
	 */
	public final int leeway;
	private static final int DEFAULT_LEEWAY = 128;
	
	/**Number of pixels the graph is expected to take up. The canvas
	 * is assumed to be square. The default value is 480 pixels
	 * in honor of the original VGA monitor standard.
	 * Note that once this value is set, it can no
	 * longer be changed.
	 */
	public final int canvas; 
	private static final int DEFAULT_CANVAS = 480;
	
	/**Maximum number of elements a component is allowed to have.
	 */
	public final int maxSize; 
	
	/**Stores which configuration will be used.
	 * The constructors enforce this to only be one of the supported
	 * configurations. 'G' is used for the Geometric configuration,
	 * 'L' for the Layered configuration, and 'S' for the Star
	 * configuration. These constants are also provided as members
	 * and are encouraged for improved maintainability.
	 * Note that once this value is set, it can no
	 * longer be changed.
	 */
	public final char configuration;
	
	/**Value for {@link #configuration} for the geometric configuration (G)
	 */
	public static final char GEOMETRIC = 'G';
	/**Value for {@link #configuration} for the layered configuration (L)
	 */
	public static final char LAYERED = 'L';
	/**Value for {@link #configuration} for the Asterisk configuration (A)
	 */
	public static final char ASTERISK = 'A';
	/**Value for {@link #configuration} for the Star configuration (S)
	 */
	public static final char STAR = 'S';
	
	/**checks if the given char is a valid configuration name
	 * 
	 * @throws IllegalArgumentException if it isn't
	 */
	private void validConfiguration(char c){
		if(c != GEOMETRIC && c != LAYERED && c != STAR && c != ASTERISK)
			throw new IllegalArgumentException("Configuration can only be "+LAYERED+
					", "+GEOMETRIC+", "+ASTERISK+", or "+STAR+", but "
					+this.configuration+" was found instead.");
	}
	
	/**Initializes the class setting the parameters to the specified values
	 * 
	 * @param angle
	 * 			Value for {@link #angle}
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param canvas
	 *			Value for {@link #canvas}
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 * @param maxSize
	 * 			Value for {@link #maxSize} 
	 */
	public NodePlacer(double angle, int leeway, int canvas, char configuration, int maxSize){
		validConfiguration(configuration);
		this.angle = angle % Math.PI;
		this.leeway = leeway;
		this.canvas = canvas;
		this.configuration = configuration;
		this.maxSize = maxSize;
	}
	
	/**Initializes the class setting the parameters to the specified values
	 * and the default canvas size of 480
	 * 
	 * @param angle
	 * 			Value for {@link #angle}. Note that if the angle is not
	 * 			in the [0, 2&pi;) range, the modulus with respect to 2&pi;  
	 * 			will be taken
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 * @param maxSize
	 * 			Value for {@link #maxSize} 
	 */
	public NodePlacer(double angle, int leeway, char configuration, int maxSize){
		validConfiguration(configuration);
		this.angle = angle % Math.PI;
		this.leeway = leeway;
		this.canvas = DEFAULT_CANVAS;
		this.configuration = configuration;
		this.maxSize = maxSize;
	}
	
	/**Initializes the class with the specified angle, leaving the leeway
	 * at the default size of 128 and the canvas at the default size of 480
	 * 
	 * @param angle
	 * 			Value for {@link #angle}. Note that if the angle is not
	 * 			in the [0, 2&pi;) range, the modulus with respect to 2&pi;  
	 * 			will be taken
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 * @param maxSize
	 * 			Value for {@link #maxSize} 
	 */
	public NodePlacer(double angle, char configuration, int maxSize){
		validConfiguration(configuration);
		this.angle = angle % (2*Math.PI);
		this.leeway = DEFAULT_LEEWAY;
		this.canvas = DEFAULT_CANVAS;
		this.configuration = configuration;
		this.maxSize = maxSize;
	}
	
	/**Initializes the class with the specified leeway, leaving the angle
	 * at the default value of zero and the canvas at the default size of 480
	 * 
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 * @param maxSize
	 * 			Value for {@link #maxSize} 
	 */
	public NodePlacer(int leeway, char configuration, int maxSize){
		validConfiguration(configuration);
		this.angle = DEFAULT_ANGLE;
		this.leeway = leeway;
		this.canvas = DEFAULT_CANVAS;
		this.configuration = configuration;
		this.maxSize = maxSize;
	}
	
	/**Creates an instance with the default values.
	 * 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 * @param maxSize
	 * 			Value for {@link #maxSize} 
	 */
	public NodePlacer(char configuration, int maxSize){
		validConfiguration(configuration);
		this.angle = DEFAULT_ANGLE;
		this.leeway = DEFAULT_LEEWAY;
		this.canvas = DEFAULT_CANVAS;
		this.configuration = configuration;
		this.maxSize = maxSize;
	}
	
	/**Computes the size of the canvas needed to plot
	 * a graph with the specified number of components and nodes.
	 * This function assumes that {@link #leeway} and {@link #configuration}
	 * have already been initialized.
	 * 
	 * 
	 * @param components
	 * 		Number of components the graph to be plotted has
	 * @return the number of pixels needed to plot this graph
	 */
	public int computeCanvas(int components){
		switch(this.configuration){
			case LAYERED:
				return this.leeway*this.maxSize;
			case GEOMETRIC:
				return (int) (this.maxSize*this.leeway * Math.sin( Math.PI/components ));
			case ASTERISK:
				return this.maxSize*this.leeway;
			case STAR:
				return (int) (this.maxSize*2*this.leeway*Math.sin(Math.PI/this.maxSize) * Math.sin( Math.PI/components ));
			default:
				//if none of the above, throw the same exception as validConfiguraiton
				//Note that all constructors should call validConfiguration, so
				//this case should never occur
				throw new IllegalStateException("Configuration can only be "+LAYERED+
						", "+GEOMETRIC+", "+ASTERISK+", or "+STAR+", but "
						+this.configuration+" was found instead.");
		}
	}
	
	/**Initializes the class computing the canvas size to fit
	 * a graph with the specified number of components and nodes
	 * using the default leeway
	 *  
	 * @param components
	 * 		Number of components the graph to be plotted has
	 * @param componentSize
	 * 		Number of nodes in the largest component
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 */
	public NodePlacer(int components, int maxSize, int leeway, double angle, char configuration){
		validConfiguration(configuration);
		this.configuration = configuration;
		this.angle = angle;
		this.leeway = leeway;
		this.maxSize = maxSize;
		this.canvas = computeCanvas(components);
	}
	
	/**Initializes the class computing the canvas size to fit
	 * a graph with the specified number of components and nodes
	 * using the default leeway
	 *  
	 * @param components
	 * 		Number of components the graph to be plotted has
	 * @param maxSize
	 * 		Number of nodes in the largest component
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 */
	public NodePlacer(int components, int maxSize, double angle, char configuration){
		validConfiguration(configuration);
		this.configuration = configuration;
		this.angle = angle;
		this.leeway = DEFAULT_LEEWAY;
		this.maxSize = maxSize;
		this.canvas = computeCanvas(components);
	}
	
	/**Initializes the class computing the canvas size to fit
	 * a graph with the specified number of components and nodes,
	 * using the default angle
	 *  
	 * @param components
	 * 		Number of components the graph to be plotted has
	 * @param maxSize
	 * 		Number of nodes in the largest component
	 * @param leeway
	 * 			Value for {@link #leeway} 
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 */
	public NodePlacer(int components, int maxSize, int leeway, char configuration){
		validConfiguration(configuration);
		this.configuration = configuration;
		this.angle = DEFAULT_ANGLE;
		this.leeway = leeway;
		this.maxSize = maxSize;
		this.canvas = computeCanvas(components);
	}
	
	/**Initializes the class computing the canvas size to fit
	 * a graph with the specified number of components and nodes.
	 * All other parameters are set to the defaults
	 * 
	 * @param components
	 * 		Number of components the graph to be plotted has
	 * @param maxSize
	 * 		Number of nodes in the largest component
	 * @param configuration
	 * 			Value for {@link #configuration} 
	 */
	public NodePlacer(int components, int maxSize, char configuration){
		validConfiguration(configuration);
		this.configuration = configuration;
		this.leeway = DEFAULT_LEEWAY;
		this.angle = DEFAULT_ANGLE;
		this.maxSize = maxSize;
		this.canvas = computeCanvas(components);
	}
	
	/**Finds the center of the canvas
	 * 
	 * @return the x,y coordinates of the center
	 */
	public Point canvasCenter(){
		return new Point(this.canvas/2, this.canvas/2);
	}
	
	/**returns the coordinates of the {@code i}th corner of an {@code n}-gon
	 * rotated {@code theta} radians
	 *  
	 * @param i
	 * 		Desired corner
	 * @param n
	 * 		Number of sides in the polygon
	 * @param theta
	 * 		Base angle of rotation
	 * @param theta
	 * 		distance the desired point should be from the origin
	 */
	private static Point geometricCorner(int i, int n, double theta, float radius){
		theta += 2*Math.PI * i/n;
		int x = Math.round((float)Math.cos(theta)*radius);
		int y = Math.round((float)Math.sin(theta)*radius);
		return new Point(x, y);
	}
	
	/**Computes the x,y coordinates at which to place the center of
	 * the given component
	 * 
	 * @param component
	 * 		component desired
	 * @param number
	 * 		Total number of components
	 * @return
	 * 		The coordinates at which to center the requested component
	 */
	public Point componentCenter(int component, int number) {
		if(this.configuration == LAYERED){
			int trueCanvasSize = Math.max(this.canvas, this.leeway*number);
			int offset = trueCanvasSize*component/number;
			int x = Math.round((float)Math.cos(this.angle-Math.PI/2)*offset);
			int y = Math.round((float)Math.sin(this.angle-Math.PI/2)*offset);
			return new Point(x, y);
		}else{
			Point center = canvasCenter();
			if(this.configuration == STAR)
				if(component == 0){
					return center;
				}else{
					component--;
					number--;
					assert component >=0;
					assert number >= 0;
					//proceed as with geometric
				}
			float circle;	//separation between components
			if(this.configuration == ASTERISK)
				circle = (int)(this.leeway*number*Math.sin( Math.PI/number ));
			else
				circle = (float)this.canvas;
			Point offset = geometricCorner(component,number, this.angle, circle);
			return new Point(center.x+offset.x, center.y+offset.y);
		}
	}

	/**Computes the offset of the desired node from the center of the
	 * desired component
	 * 
	 * @param component
	 * 		Number of the component the desired node is in
	 * @param node
	 * 		Number of the desired node within the component
	 * @param number
	 * 		Total number of components
	 * @param size
	 * 		Total number of nodes in this component
	 * @return
	 * 		An ordered pair indicating the 2D distance from the component center
	 * 		that this node will have
	 */
	public Point nodeOffset(int component, int node, int number, int size) {
		if(this.configuration == STAR){
			return geometricCorner(node, size, this.angle, (int)(2*this.leeway*Math.sin(Math.PI/this.maxSize)));
		}else{
			double angle = this.angle;
			if(this.configuration == ASTERISK){
				angle += 2*Math.PI * component/number;
				//proceed as with layered
			}
			if(this.configuration == GEOMETRIC){
				angle += 2*Math.PI * component/number - Math.PI/number;
				//proceed as with layered
			}
			
			int componentSize = this.maxSize*this.leeway;//size of the lines to draw the nodes along
			if(this.configuration != ASTERISK)
				componentSize = Math.max((int)(this.canvas/Math.sin( Math.PI/number )), componentSize);
			
			//move to new function placeAlongLine( length of line (componentsize) , angle);
			int offset = componentSize*node/size; //offset with respect to corner
			if(this.configuration == GEOMETRIC){
				//convert to offset with respect to center
				offset -= componentSize/2 + 2*this.leeway;
			}
			int x = Math.round((float)Math.cos(angle)*offset);
			int y = Math.round((float)Math.sin(angle)*offset);
			return new Point(x, y);
		}
	}
	
	
	/**Computes the x,y coordinates at which to position the 
	 * specified node for the desired {@link #configuration}
	 * 
	 * @param component
	 * 		Number of the component the node to be placed is in
	 * @param node
	 * 		Number of the desired node within the component
	 * @param number
	 * 		Total number of components
	 * @param size
	 * 		Total number of nodes in this component
	 * @return
	 * 		The coordinate at which to place the requested node
	 * 
	 * @throws IllegalArgumentException
	 * 		if {@code node} is greater than{@code size}, 
	 * 		if {@code size} is greater than {@link #maxSize},
	 * 		or if {@code component} is greater than {@code number}
	 */
	public Point position(int component, int node, int number, int size)
			throws IllegalArgumentException{
		if(node>size)
			throw new IllegalArgumentException("Cannot place the "+node+"th node "
					+"of a component of size "+size);
		if(size>this.maxSize)
			throw new IllegalArgumentException("NodePlacer instance was created with "
					+"a maximum component size of "+maxSize+", yet a component "
					+"of size "+size+" was attempted to be placed");
		if(component>number)
			throw new IllegalArgumentException("Cannot place the "+component+"th "
					+"component of a set of "+number+" components");
		Point componentCenter = componentCenter(component, number);
		System.out.println("Component "+component+" of "+number+" is centered at "+componentCenter);
		Point offset = nodeOffset(component, node, number, size);
		System.out.println("Node "+node+" of "+size+" offset by "+offset);
		return new Point(componentCenter.x+offset.x, componentCenter.y+offset.y);
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.err.println("What are you doing? The main's not ready yet!");
	}

}
