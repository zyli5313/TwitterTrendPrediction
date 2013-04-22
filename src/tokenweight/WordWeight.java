package tokenweight;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class WordWeight {
	public static class WeightMapper extends Mapper<LongWritable, Text, Text, Text>{
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	String[] keyValue = value.toString().split("\t");
		    context.write(new Text(keyValue[0]), new Text(keyValue[1]));
	    }
	}
	
	public static class WeightReducer extends Reducer<Text,Text,Text,DoubleWritable>{
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Iterator<Text> iter = values.iterator();
			String value1 = iter.next().toString();
			String value2 = null;
			if(value1.startsWith("~")) {
				value1 = value1.substring(1);
				value2 = iter.next().toString();
			}else{
				if(iter.hasNext()){
					value1 = iter.next().toString().substring(1);
					value2 = value1;
				}else{
					return;
				}
			}
			double weight = Math.pow((Double.parseDouble(value1)-Double.parseDouble(value2)), 2)/Double.parseDouble(value2);
			context.write(key, new DoubleWritable(weight));
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();    
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 4) {
	    	System.err.println("Usage: WordWeight <in1> <in2> <out> <reducer>");
	    	System.exit(4);
	    }
	    Job job = new Job(conf, "WordWeight");
	    job.setJarByClass(WordWeight.class);
	    job.setMapperClass(WeightMapper.class);
	    job.setReducerClass(WeightReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(DoubleWritable.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[3]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
