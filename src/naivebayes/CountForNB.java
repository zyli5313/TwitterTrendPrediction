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

public class CountForNB {
	public static class CountMapper extends Mapper<LongWritable, Text, Text, IntWritable>{
		private final static IntWritable one = new IntWritable(1);
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	String[] keyValue = value.toString().split("\t");
	    	String[] classes = keyValue[0].split(",");
	    	for(String s:classes){
	    		context.write(new Text("!"+s),one);
			}
	    	context.write(new Text("#totalY"),one);
	    	Vector<String> tokens = tokenizeDoc(keyValue[1]);
	    	for(String s:classes){
	    		int count = 0;
	    		for(String str:tokens){
	    			String keyPair = str+":"+s;
	    			context.write(new Text(keyPair),one);
	    			count++;
	    		}
	    		context.write(new Text("%"+s),new IntWritable(count));
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
	}
	
	public static class CountCombiner extends Reducer<Text,IntWritable,Text,IntWritable>{
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
	public static class CountReducer extends Reducer<Text,IntWritable,Text,Text> {
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			if(key.toString().contains(":")){
				context.write(new Text(key.toString().split(":")[0]), new Text(key+"="+sum));
			}else{
				context.write(key, new Text(String.valueOf(sum)));
			}
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) {
	      System.err.println("Usage: CountForNB <in> <out> <reducer>");
	      System.exit(3);
	    }
	    Job job = new Job(conf, "CountForNB");
	    job.setJarByClass(CountForNB.class);
	    job.setMapperClass(CountMapper.class);
	    job.setCombinerClass(CountCombiner.class);
	    job.setReducerClass(CountReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(IntWritable.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[2]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}