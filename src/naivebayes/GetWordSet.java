package naivebayes;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

public class GetWordSet {
	HashSet<String> word = new HashSet<String>();
	public void setWord(String doc){
		Vector<String> str= tokenizeDoc(doc);
		for(String s:str){
			word.add(s);
		}
	}
	public Vector<String> tokenizeDoc(String cur_doc) {
		String[] words = cur_doc.split("\\s+");
		Vector<String> tokens = new Vector<String>();
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "");
			if (words[i].length() > 0) {
				tokens.add(words[i]);
			}
		}
		return tokens;
	}
	public void printResult(String file) throws IOException{
		BufferedWriter br = new BufferedWriter(new FileWriter(file));
		Iterator<String> it = word.iterator();
		while(it.hasNext()){
			String s = it.next();
			br.write(s+"\t"+"#\n");
		}
		br.flush();
		br.close();
	}
	public static void main(String[] args) throws IOException{
		GetWordSet gws = new GetWordSet();
		String file = args[0];
		BufferedReader br = new BufferedReader(new FileReader(file));
		while(br.ready()){
			String doc = br.readLine();		
			gws.setWord(doc);
		}
		gws.printResult(args[1]);
		
		br.close();
	}
}
