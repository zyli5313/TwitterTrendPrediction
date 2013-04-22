package naivebayes;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class FilterWordCount {
	public static class FilterMapper extends Mapper<LongWritable, Text, Text, Text>{
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	context.write(new Text(value.toString().split("\t")[0]),new Text(value.toString().split("\t")[1]));
	    	if(!value.toString().contains("!")&&!value.toString().contains("%")&&!value.toString().contains("#"))
	    		context.write(new Text("#totalV"), new Text("1"));
	    }
	}
	public static class FilterReducer extends Reducer<Text,Text,Text,Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			if(key.toString().equals("#totalV")){
				int sum = 0;
				for (Text val : values) {
					sum = sum +1;
				}
				context.write(key, new Text(String.valueOf(sum)));
			}else if(!key.toString().contains("!")&&!key.toString().contains("%")&&!key.toString().contains("#")){
				String count = "";
				int i = 0;
				for (Text val : values) {
					if(!val.toString().equals("#")) count = val.toString();
					i++;
				}
				if(i==2) context.write(key, new Text(count.toString()));
			}else{
				context.write(key, new Text(values.iterator().next().toString()));
			}
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 4) {
	      System.err.println("Usage: FilterWordCount <in> <wordset> <out> <reducer>");
	      System.exit(4);
	    }
	    Job job = new Job(conf, "FilterWordCount");
	    job.setJarByClass(FilterWordCount.class);
	    job.setMapperClass(FilterMapper.class);
	    job.setReducerClass(FilterReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[3]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
