package naivebayes;

import java.io.IOException;

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

public class MergeWordCount {
	public static class MergeMapper extends Mapper<LongWritable, Text, Text, Text>{
	    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
	    	context.write(new Text(value.toString().split("\t")[0]),new Text(value.toString().split("\t")[1]));
	    }
	}
	public static class MergeReducer extends Reducer<Text,Text,Text,Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			StringBuffer res = new StringBuffer();
			int count = 0;
			for (Text val : values) {
				res.append(val.toString()).append(",");
				count++;
			}
			res.deleteCharAt(res.length()-1);
			context.write(key, new Text(res.toString()));
			if(count>1) context.write(new Text("#totalDic"), new Text("1"));
		}
	}
	public static void main(String[] args) throws Exception {
	    Configuration conf = new Configuration();
	    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) {
	      System.err.println("Usage: MergeWordCount <in> <out> <reducer>");
	      System.exit(3);
	    }
	    Job job = new Job(conf, "MergeWordCount");
	    job.setJarByClass(MergeWordCount.class);
	    job.setMapperClass(MergeMapper.class);
	    job.setReducerClass(MergeReducer.class);
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
	    job.setNumReduceTasks(Integer.parseInt(otherArgs[2]));
	    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}
