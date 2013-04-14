package naivebayes;

import java.io.IOException;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.json.JSONException;
import org.json.JSONObject;

import com.aliasi.tokenizer.*;

public class BackGroundWordCount {
	public static class CountMapper extends Mapper<LongWritable, Text, Text, IntWritable>{
		private final static IntWritable one = new IntWritable(1);
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	try{
	    		String[] keyValue = value.toString().split("\t");
		    	String jsonLine = keyValue[2];
		    	JSONObject json = new JSONObject(jsonLine);
		        String text = json.getString("text");        
		    	Vector<String> tokens = tokenizeDoc(text);
		    	int count = 0;
		    	for(String str:tokens){
		    		context.write(new Text(str),one);
		    		count++;
		    	}
		    	context.write(new Text("#Total"),new IntWritable(count));
	    	}catch(JSONException je){
	    		//do nothing
	    	}
	    		    	
	    }
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
	}
	
	public static class CountReducer extends Reducer<Text,IntWritable,Text,IntWritable>{
		private IntWritable result = new IntWritable();
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) {
	      System.err.println("Usage: CountForNB <in> <out> <reducer>");
	      System.exit(3);
	    }
	    Job job = new Job(conf, "BackGroundWordCount");
	    job.setJarByClass(BackGroundWordCount.class);
	    job.setMapperClass(CountMapper.class);
	    job.setCombinerClass(CountReducer.class);
	    job.setReducerClass(CountReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(IntWritable.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(IntWritable.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[2]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
