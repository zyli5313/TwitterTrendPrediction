package naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import org.json.JSONException;
import org.json.JSONObject;

import tokenweight.NormalizedTokenizerFactory;

import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.LowerCaseTokenizerFactory;
import com.aliasi.tokenizer.PorterStemmerTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;


public class TestNB {
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
	public void processTestData(String file) throws IOException{
		int count = 0;
		BufferedReader br = new BufferedReader(new FileReader(file));
		String tag = null;
		String l = null;
		StringBuffer sb = new StringBuffer();
		int[] tp = new int[6];
		int[] fp = new int[6];
		int[] tn = new int[6];
		int[] fn = new int[6];
		double precision = 0;
		double recall = 0;
		double F = 0;
		while(br.ready()){
			count++;
			String doc = br.readLine();	
			String[] attr = doc.split("\t");
			String hashTag = attr[0];
			String label = attr[1];
			String jsonLine = attr[2];
			String text = null;
			try{
				JSONObject json = new JSONObject(jsonLine);
		        text = json.getString("text");
			}catch(JSONException je){
				text = "";
			}
	        if(tag!=null&&!hashTag.equals(tag)){
	        	String c = testDoc(sb.toString().trim());
	        	System.out.println(tag+" "+c+" "+l);
	        	sb = new StringBuffer();
	        	if(c.equals(l))  tp[Integer.parseInt(c)-1]++;
	        	else {
	        		fp[Integer.parseInt(c)-1]++;
	        		fn[Integer.parseInt(l)-1]++;
	        	}    	
	        	count = 0;
	        }
	        tag = hashTag;
	        l = label;
	        sb.append(text+" ");
		}
    	String c = testDoc(sb.toString().trim());
    	if(c.equals(l))  tp[Integer.parseInt(c)]++;
    	else {
    		fp[Integer.parseInt(c)]++;
    		fn[Integer.parseInt(l)]++;
    	}   
		br.close();
		for(int i=0;i<6;i++){
			precision = precision + (double)tp[i]/(tp[i]+fp[i]);
			recall = recall + (double)tp[i]/(tp[i]+fn[i]);
			F = 2*precision*recall/(precision+recall);
		}
		System.out.println("Precision:"+precision/6);
		System.out.println("Recall:"+recall/6);
		System.out.println("F:"+F/6);
	}
	public String testDoc(String text){
		if(text!=null){
			Vector<String> tokens = tokenizeDoc(text);
			Iterator<String> it = countY.keySet().iterator();
			double p = Double.NEGATIVE_INFINITY;
			String maxClass = "";
			while(it.hasNext()){
				String c = (String)it.next();
				double prob = Math.log((double)(1+countY.get(c))/(double)(countY.size()+totalY));
				//System.out.println(prob);
				for(String s:tokens){
					prob = prob + calProb(c,s);
				}
				//System.out.println(prob);
				if(prob>p){
					p = prob;
					maxClass = c;
				}
				prob = 0;
			}
			return maxClass;		
		}else{
			return null;
		}
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
		double count = 0;
		double time = 1;
		if(countYToken.containsKey(token+":"+clas)){
			count = countYToken.get(token+":"+clas);
			//time = 1+100000*weight.get(token);
			//System.out.println(totalVoca);
			count = count * time;
		}
		return Math.log((double)(count+1)/(double)(countToken.get(clas)+totalVoca));
	}
	public static void main(String[] args) throws IOException{
		TestNB nb = new TestNB();
		//Read Training Model
		BufferedReader br = new BufferedReader(new FileReader(args[0]));
		while(br.ready()){
			String doc = br.readLine();		
			nb.processRecord(doc);
		}
		br.close();
		//Read Weight Data
		
		for(int i=1;i<7;i++){
			br = new BufferedReader(new FileReader(args[2]+i));
			while(br.ready()){
				String record = br.readLine();
				String[] count = record.split("\t");
				if(nb.weight.containsKey(count[0])){
					nb.weight.put(count[0],Double.parseDouble(count[1])+nb.weight.get(count[0]));
				}else{
					nb.weight.put(count[0],Double.parseDouble(count[1]));
				}
				
			}
			br.close();
		}
		
		//Read Test Data
		nb.processTestData(args[1]);
	}
}
