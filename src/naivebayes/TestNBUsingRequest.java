package naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import tokenweight.NormalizedTokenizerFactory;

import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.LowerCaseTokenizerFactory;
import com.aliasi.tokenizer.PorterStemmerTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;


public class TestNBUsingRequest {
	//The Weight of Different Token
	private HashMap<String,Double> weight = new HashMap<String,Double>();
	//The number of training instances of class Y
	private HashMap<String,Integer> countY = new HashMap<String,Integer>();   
	//Total number of tokens for document with lable Y
	private HashMap<String,Integer> countToken = new HashMap<String,Integer>();
	//Number of times token w appears in a document with lable Y
	private HashMap<String, Integer> countYToken = new HashMap<String,Integer>();
	//Number of instances
	private int totalY;
	private int totalVoca;
	private int totalTest = 0;
	private double totalRight = 0;
	private double precision = 0;
	
	public void processRecord(String record){
		String[] mapping = record.split("\t");
		if(mapping[0].charAt(0)=='!'){
			countY.put(mapping[0].substring(1), Integer.parseInt(mapping[1]));
			return;
		}else if(mapping[0].charAt(0)=='%'){
			countToken.put(mapping[0].substring(1), Integer.parseInt(mapping[1]));
			return;
		}else if(mapping[0].equals("#totalInstance")) {
			totalY = Integer.parseInt(mapping[1]);
			return;
		}else if(mapping[0].equals("#totalDic")) {
			totalVoca = Integer.parseInt(mapping[1]);
			return;
		}else{		
			String[] pair = mapping[1].split(",");
			for(String p:pair){			
				String[] keyvalue = p.split("=");
				countYToken.put(keyvalue[0], Integer.valueOf(keyvalue[1]));
			}						
		}
		return;
	}
	public void testDoc(String text){
		ArrayList<String> cla = new ArrayList<String>();
		if(text!=null){
			//split class token with document
			String[] attr = text.split("\t");
			if(attr.length>1){
				//deal with class
				String[] classes = attr[0].split(",");
				for(String s:classes){
					cla.add(s);
				}
				//deal with tokens
				Vector<String> tokens = tokenizeDoc(attr[1]);
				Iterator<String> it = countY.keySet().iterator();
				double p = -1000000;
				String maxClass = "";
				while(it.hasNext()){
					String c = (String)it.next();
					double prob = Math.log((double)(1+countY.get(c))/(double)(countY.size()+totalY));
					for(String s:tokens){
						prob = prob + calProb(c,s);
					}
					if(prob>p){
						p = prob;
						maxClass = c;
					}
					prob = 0;
				}
				totalTest++;
				if(cla.contains(maxClass)) totalRight++;
				String result;
				result = "["+attr[0]+"]"+"\t"+maxClass+"\t"+p;
				System.out.println(result);
			}			
		}
		//countYToken.clear();
	}
	/*
	 * Tokenize the document into tokens
	 */
	public Vector<String> tokenizeDoc(String tweet) {
		Vector<String> tokens = new Vector<String>();
		TokenizerFactory tokFactory = new NormalizedTokenizerFactory();
    	tokFactory = new LowerCaseTokenizerFactory(tokFactory);
    	tokFactory = new EnglishStopTokenizerFactory(tokFactory);
    	tokFactory = new PorterStemmerTokenizerFactory(tokFactory);
    	char[] chars = tweet.toCharArray();
    	Tokenizer tokenizer 
    	    = tokFactory.tokenizer(chars,0,chars.length);
    	String token;
		while ((token = tokenizer.nextToken()) != null) {
			token = token.toLowerCase();		
		    tokens.add(token);
		}
		return tokens;
	}
	public double calProb(String clas,String token){
		int count = 0;
		if(countYToken.containsKey(token+":"+clas)){
			count = countYToken.get(token+":"+clas);
		}
		return Math.log((double)(count+1)/(double)(countToken.get(clas)+totalVoca));
	}
	public static void main(String[] args) throws IOException{
		TestNBUsingRequest nb = new TestNBUsingRequest();
		//Read Training Model
		BufferedReader br = new BufferedReader(new FileReader(args[0]));
		while(br.ready()){
			String doc = br.readLine();		
			nb.processRecord(doc);
		}
		br.close();
		//Read Weight Data
		for(int i=0;i<6;i++){
			br = new BufferedReader(new FileReader(args[1]+i));
			while(br.ready()){
				String record = br.readLine();
				String[] count = record.split("\t");
				nb.weight.put(count[0]+":"+i,Double.parseDouble(count[1]));
			}
			br.close();
		}
		//Read Test Data
		br = new BufferedReader(new FileReader(args[2]));
		while(br.ready()){
			String doc = br.readLine();		
			nb.testDoc(doc);
		}
		br.close();
		nb.precision = nb.totalRight/nb.totalTest;
		System.out.println("Percent Correct: "+nb.precision);
	}
}
