package ve.usb.testMain;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.SortedSet;

class A{
	private int value;
	
	public A(){
		this.value = 0;
	}
	
	public static void main(String[] args){
		System.out.println(
				"* * * * * * * * * * * * * *\n" +
				"* 'A' class test program  *\n" +
				"* * * * * * * * * * * * * *\n" +
				"This program lets you test the A class by creating an\n" +
				"instance of A and allowing you to perform any of the\n" +
				"operations of the class on it.");
		A a = new A();
		
		Map<String, Callable> m = TestProgram.baseMap();
		SortedSet<String> help  = TestProgram.baseHelp();
		
		String[] subtractArgs = new String[1];
		subtractArgs[0] = "<int>";
		TestProgram.put(m, help, "subtract", new subtractCall(a), subtractArgs);
		
		TestProgram.put(m, help, "help", new TestProgram.defaultHelp(help), new String[0]);
		
		TestProgram p = new TestProgram(m);
		
		BufferedReader stdIn = new BufferedReader(new InputStreamReader(System.in));
		while(true){
			try{
				System.out.print("\n~:");
				System.out.println( p.parseInput( stdIn.readLine() ) );
			}catch(Exception e){
				e.printStackTrace();
				System.err.println();
			}
		}
	}
	
	public int subtract(int ammount){
		this.value -= ammount;
		return value;
	}
	
	private static class subtractCall implements Callable{
		A a;
		
		subtractCall(A a){
			this.a = a;
		}
		
		public String call(String[] args){
			return ""+a.subtract( Integer.parseInt(args[0]) );
		}
		public int argNum(){
			return 1;
		}
	}
}
