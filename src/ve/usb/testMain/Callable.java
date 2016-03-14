package ve.usb.testMain;

/**Defines an interface for classes that have a single functional function to be called: {@link #call}
 * More specifically, this interface is designed for classes whose purpose is to provide a wrapper
 * for a function, allowing said function to be passed as a parameter.
 * <p>
 * Implementing classes wrapping functions that receive arguments must obtain them from an array of Strings,
 * converting them to the appropriate types before passing them onto the function they are wrapping.
 * Similarly, what the function returns must be converted to String.
 * <p>
 * To allow callers to check they have the right number of arguments, the interface provides
 * them with the {@link #argNum} method. Implementing classes are allowed to perform said
 * check themselves (such as when the function has overload) if, and only if,
 * they return an {@code argNum} of {@code -1}
 * 
 * @author Fernando J. Torre
 * @since  2012-02-13
 */
public interface Callable{
	/**Calls the function of this class
	 * 
	 * @param args the arguments for the function as {@code String}s
	 * 
	 * @return the return value for this function, or an empty string if the function
	 *         returns {@code void}
	 * 
	 * @throws Exception if the arguments received cause the function to throw one.
	 */
	public String call(String[] args) throws Exception;
	
	/**Returns the number of arguments the function of this class receives.
	 * 
	 * @return the number of arguments, or {@code -1} if the implementing class would prefer
	 *         the calling class not to know.
	 */
	public int argNum();
}
