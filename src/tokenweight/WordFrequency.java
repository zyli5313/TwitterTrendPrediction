package tokenweight;

import java.io.IOException;

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

public class WordFrequency {
	public static class CountMapper extends Mapper<LongWritable, Text, Text, DoubleWritable>{
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	String[] keyValue = value.toString().split("\t");
	    	if(keyValue[0].contains("#")) return;
		    double count = Double.parseDouble(keyValue[1]);
		    Configuration mapconf = context.getConfiguration();
            int totalCount = Integer.parseInt(mapconf.get("totalCount"));
		    count = count/totalCount;
		    context.write(new Text(keyValue[0]), new DoubleWritable(count));
	    }
	}
	
	public static class CountReducer extends Reducer<Text,DoubleWritable,Text,Text>{
		public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
			Configuration mapconf = context.getConfiguration();
			if(mapconf.get("bg").equals("bg")) 
				context.write(key, new Text(values.iterator().next().toString()));
			else if(mapconf.get("bg").equals("fg")) 
				context.write(key, new Text("~"+values.iterator().next().toString()));
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();    
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 5) {
	      System.err.println("Usage: WordFrequency <in> <out> <total> <bg or fg> <reducer>");
	      System.exit(5);
	    }
	    conf.set("totalCount", otherArgs[2]);
	    conf.set("bg", otherArgs[3]);
	    Job job = new Job(conf, "WordFrequency");
	    job.setJarByClass(WordFrequency.class);
	    job.setMapperClass(CountMapper.class);
	    job.setReducerClass(CountReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(DoubleWritable.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[4]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
