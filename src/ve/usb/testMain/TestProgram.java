package ve.usb.testMain;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

/**Gives a general structure for making {@code main} test programs.
 * Test programs should provide a prompt (by convention, a tilde and a colon &ldquo;{@code ~:}&rdquo;)
 * wherein the user should be allowed to type any function of that class where the
 * parameters can be passed as Strings. Generic classes should be instanced to use
 * strings to allow this. 
 * <p>
 * So that this class knows how to call the functions of the class it is testing,
 * it must be constructed with a {@link Map} of {@link Callable}s
 * (mapping the function's name to its respective {@code Callable}),
 * to which it should only have to pass each of the parameters input by the user.
 * It is the responsibility of the {@code Callable}s to convert the {@code String}s
 * into {@code int}s (or any other appropriate type) when necessary before performing
 * the function call, and to implement function overload (i.e. two functions with the same name,
 * even if the number of parameters differs, will be accessed by this class via the same {@link Callable}).
 * This class will always {@code trim} the {@code String}s before passing them onto the {@code Callable}.
 * <p>
 * <strong>Calling classes are hereby required include in the map the methods &ldquo;{@code help()}&rdquo;
 * &ndash;which must list the methods in the {@code Map}&ndash; and &ldquo;{@code quit()}&rdquo; &ndash;which
 * must cause the program to exit.</strong> By convention, the list returned by {@code help()}
 * should be in alphabetical order.
 * <p>
 * A sample class {@code A}, that has only one method, and uses this class to create it's test program, follows:
 * <blockquote>
 * 	<code>
 * 	class A{<br/>
 * 	&nbsp; &nbsp; private int value;<br/>
 * 	&nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; public A(){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; this.value = 0;<br/>
 * 	&nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; public static void main(String[] args){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; System.out.println(<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "* * * * * * * * * * * * * *\n" +<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "* 'A' class test program &nbsp;*\n" +<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "* * * * * * * * * * * * * *\n" +<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "This program lets you test the A class by creating an\n" +<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "instance of A and allowing you to perform any of the\n" +<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; "operations of the class on it.");<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; A a = new A();<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; Map&lt;String, Callable&gt; m = TestProgram.baseMap();<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; SortedSet&lt;String&gt; help &nbsp;= TestProgram.baseHelp();<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; String[] subtractArgs = new String[1];<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; subtractArgs[0] = "&lt;int&gt;";<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; TestProgram.put(m, help, "subtract", new subtractCall(a), subtractArgs);<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; TestProgram.put(m, help, "help", new TestProgram.defaultHelp(help), new String[0]);<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; TestProgram p = new TestProgram(m);<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in));<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; while(true){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; try{<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; System.out.print("~:");<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; System.out.println( p.parseInput( stdIn.readLine() ) );<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; }catch(Exception e){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; e.printStackTrace();<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; System.err.flush();<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; public int subtract(int amount){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; this.value -= amount;<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; return value;<br/>
 * 	&nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; private static class subtractCall implements Callable{<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; A a;<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; subtractCall(A a){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; this.a = a;<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; <br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; public String call(String[] args){<br/>
 * 	&nbsp; &nbsp; &#9&nbsp; &nbsp; return a.subtract( Integer.parseInt(args[0]) );<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; public int argNum(){<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; return 1;<br/>
 * 	&nbsp; &nbsp; &nbsp; &nbsp; }<br/>
 * 	&nbsp; &nbsp; }<br/>
 * 	}
 * 	</code>
 * </blockquote>
 * @author Fernando J. Torre
 * @since  2012-02-13
 */
public final class TestProgram {
	
	Map<String, Callable> m;
	
	/**Creates a map of {@link Callable}s with the call to "{@code quit()}" already implemented
	 */
	public static Map<String, Callable> baseMap(){
		Map<String, Callable> out = new HashMap<String, Callable>();
		out.put("quit", new Quit());
		return out;
	}
	
	/**Wrapper for {@link System#exit} */
	private static class Quit implements Callable{
		Quit(){}
		
		public String call(String[] args){
			System.exit(0);
			return "Exiting failed. Please terminate using Ctrl-C";
		}
		
		public int argNum(){
			return 0;
		}
	}
	
	public TestProgram(Map<String, Callable> map) throws IllegalArgumentException{
		if(!map.containsKey("help"))
			throw new IllegalArgumentException("Maps must contain a \"help\" function");
		if(!map.containsKey("quit"))
			throw new IllegalArgumentException("Maps must contain a \"quit\" function");
		this.m = map;
	}
	
	/**Processes the user's input trying to identify which of the functions
	 * in the map the user has typed in, and passing the arguments to the appropriate callable
	 * 
	 * @param input a string typed by the user
	 * 
	 * @return whatever the function called returns. It is the responsibility of the caller to
	 *         show this to the user through the appropriate output mechanism.
	 * 
	 * @throws IllegalArgumentException if the user wrote something that is not a function
	 *         in the map, is a function called with the wrong number of parameters,
	 *         or the called function generated an {@code IllegalArgumentException}
	 * @throws Exception if the called functions generate any other kind of exception
	 */
	public String parseInput(String input) throws Exception{
		//check it's formatted as "name(args)", like a function call should be
		String[] userInput = input.split("\\)");
		if(userInput.length != 1){
			throw new IllegalArgumentException("Only one function may be called at a time.\n" +
					"\tPlease do not write anything after the closing parenthesis,\n" +
					"\tor include parentheses in the strings you want to pass as arguments.");
		}
		userInput= userInput[0].split("\\(");
		if(userInput.length > 2){
			throw new IllegalArgumentException("Function calls must have exactly one set of parentheses.\n" +
					"\tPlease make sure the strings you are passing include no parentheses");
		}
		assert userInput.length == 2;
		//userInput[0] should now contain the name of the function, and userInput[1] all the stuff in parentheses
		
		
		//identify which function the user has input
		String functionName = userInput[0];
		Callable f = m.get(functionName);
		if(f==null)
			throw new IllegalArgumentException("The function \""+functionName+"\" is not recognized by this program\n" +
					"\tFor the list of functions the program recognizes, type in \"help()\"");
		
		//check if the user gave the correct number of arguments, if f allows this
		if(f.argNum() >= 0){
			if(userInput.length == 1){
				//userInput contains only function name and/or the function name followed by empty parentheses
				if(f.argNum() == 0)
					return f.call(new String[0]);
				else
					throw stdErr(functionName, ""+f.argNum(), 0);
			}else{
				userInput = divideIntoArgs(userInput[1]);	//userInput now stores the contents of the parentheses, separated into its arguments
				if(	userInput.length == f.argNum()	//it's the correct number of arguments
				   	|| (f.argNum() == 0 && userInput.length == 1 && userInput[0].trim().equals(""))	//the only argument contains nothing but whitespace
				  )
				{
					return f.call(userInput);
				}else{
					throw stdErr(functionName, ""+f.argNum(), userInput.length);
				}
			}
		}else{
			//f does not allow us to check
			if(userInput.length==1){
				//userInput contains only function name and/or the function name followed by empty parentheses
				return f.call(new String[0]);
			}else{
				userInput = divideIntoArgs(userInput[1]);	//userInput now stores the contents of the parentheses, separated into its arguments
				return f.call(userInput);
			}
		}
	}
	
	/**Prints an error message explaining that the number of arguments received was not the number expected.
	 * This function is made public to allow the {@link Callable}s that prefer to handle for themselves
	 * report errors in the number of arguments in a uniform way.
	 * 
	 * @param function The name of the function to be reported in the error message
	 * @param expected The number of arguments the function expected to received
	 * @param found
	 * @return an {@code IllegalArgumentException} exception with the message
	 *         {@code The function }&lt;function&gt;{@code must receive }&lt;expected&gt;{@code arguments. }&lt;found&gt;{@code were found}
	 */
	public static IllegalArgumentException stdErr(String function, String expected, int found){
		return new IllegalArgumentException("The function \""+function+"\" must receive "+expected+" arguments. " +
				found+" were found.");
	}
	
	/**splits a comma-separated list into its individual components. Each component's whitespace is {@link java.lang.String.trim}med
	 */
	protected static String[] divideIntoArgs(String s){
		String[] out = s.split(",");
		for(int i = out.length-1; i>=0; i--)
			out[i] = out[i].trim();
		return out;
	}
	
	/**Offers a simple implementation of the &ldquo;help&rdquo; method required by the constructor.
	 * When called, this class will iterate through the sorted set given
	 */
	public static class defaultHelp implements Callable{
		SortedSet<String> functions;
		
		public defaultHelp(SortedSet<String> functions){
			this.functions = functions;
		}
		
		public int argNum(){
			return 0;
		}
		
		public String call(String[] args){
			String out = "The following is the list of functions the \"main\" program can process:\n";
			for(Iterator<String> I = this.functions.iterator(); I.hasNext();){
				out += "\t"+I.next()+"\n";
			}
			return out;
		}
	}
	
	/**Creates a {@code SortedSet} containing only &ldquo;{@code quit()}&rdquo;.
	 * This method is created to mirror the functionality of {@link #baseMap} for
	 * {@link TestProgram.defaultHelp}
	 */
	public static SortedSet<String> baseHelp(){
		SortedSet<String> out = new TreeSet<String>();
		out.add("quit()");
		return out;
	}
	
	/**Adds the given function to both the given {@code Map} and the given {@code SortedSet}.
	 * This method is intended to make the use of {@link TestProgram.defaultHelp} easier by
	 * performing the appropriate {@code Map.put} call, and creating the elements of the sorted set in the form
	 * <blockquote>
	 * 	{@code functionName(functonArg}<sub>0</sub>, {@code functonArg}<sub>1</sub>, {@code functonArg}<sub>2</sub>, &hellip;{@code )}
	 * </blockquote>
	 * in a single call.
	 * <p>
	 * The method requires {@code functionCall.argNum} and {@code functionArgs.length} to match if {@code functionCall.argNum}
	 * is not -1. Do not use this method if you require them to differ for any reason.
	 * <p>
	 * This method does not check for any sort of correspondence between the existing elements of {@code map} and {@code functions}
	 * 
	 * @param map          The map, as required by {@link #TestProgram(Map)}
	 * @param functions    The set of functions, as required by {@link TestProgram.defaultHelp} 
	 * @param functionName The name of the function to be added
	 * @param functionCall The {@code Callable} that performs the call of the function to be added
	 * @param functionArgs The list of arguments of the function to be added
	 * 
	 * @throws AssertionError if, when {@code functionCall.argNum() == -1}, it turns out that {@code functionCall.argNum() != functionArgs.length}
	 */
	public static void put(Map<String, Callable> map, SortedSet<String> functions, String functionName, Callable functionCall, String[] functionArgs) throws AssertionError{
		assert (functionCall.argNum() == functionArgs.length) || (functionCall.argNum() == -1);
		
		map.remove(functionName);
		map.put(functionName, functionCall);
		
		functions.add(buildFunction(functionName, functionArgs));
	}
	
	/**Turns the function name and argument names given into a single
	 * string of the form
	 * <blockquote>
	 * 	{@code functionName(functonArg<sub>0</sub>, functonArg<sub>1</sub>, functonArg<sub>2</sub>, &hellip;)}
	 * </blockquote>
	 * 
	 * @param functionName the name of the function
	 * @param functionArgs the names of the arguments of the function 
	 */
	private static String buildFunction(String functionName, String[] functionArgs){
		String declaration = functionName.trim()+"(";
		for(int i = 0; i<functionArgs.length; i++){
			declaration += functionArgs[i];
			if(i < functionArgs.length-1)
				declaration += ", ";
		}
		return declaration+")";
	}
	
	/**Adds the given function to the given map and to the given
	 * map's &ldquo;help&rdquo; function, provided it is a {@link TestProgram.defaultHelp} instance.
	 * If the map does <em>not</em> map "help" to a {@code defaultHelp} instance,
	 * the function should be added, and the help function replaced,
	 * using {@code Map.put},
	 * <p>
	 * The method requires {@code functionCall.argNum} and {@code functionArgs.length} to match if {@code functionCall.argNum}
	 * is not -1. Do not use this method if you require them to differ for any reason.
	 * <p>
	 * This method assumes the help function contained by the map is a correct {@link defaultHelp} instance.
	 * This means it assumes that another function by the given name doesn't already exist.
	 * If one does exist, but the arguments are different, the "help" function
	 * will print both, the previously existing version, and the new version.
	 * This is done so mainly for efficiency, and because the existing version is assumed
	 * to be valid (say, belonging to a superclass).
	 * If classes require the existing version not to be printed, they must remove it themselves.
	 * 
	 * @param map          The map, as required by {@link #TestProgram(Map)}
	 * @param functionName The name of the function to be added
	 * @param functionCall The {@code Callable} that performs the call of the function to be added
	 * @param functionArgs The list of arguments of the function to be added
	 * 
	 * @throws IllegalArgumentException If the given map does not contain a "help" function
	 * @throws ClassCastException       If the map does contain a "help" function, but it
	 *                                  is not a {@code defaultHelp} instance
	 * @throws AssertionError if, when {@code functionCall.argNum() == -1}, it turns out that {@code functionCall.argNum() != functionArgs.length}
	 */
	public static void put(Map<String, Callable> map, String functionName, Callable functionCall, String[] functionArgs) throws IllegalArgumentException, ClassCastException, AssertionError{
		defaultHelp help = (defaultHelp) map.get("help");
		if(help == null)
			throw new IllegalArgumentException("the map provided does not contain a help call");
		assert (functionCall.argNum() == functionArgs.length) || (functionCall.argNum() == -1);
		map.remove(functionName);
		map.put(functionName, functionCall);
	}
	
}
