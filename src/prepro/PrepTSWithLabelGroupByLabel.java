package prepro;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.mapred.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.json.JSONObject;

public class PrepTSWithLabelGroupByLabel extends Configured implements Tool {

  private Path inPath, out1Path;

  public String htagPath, lbPath;

  private int nreducers = 3;

  /**
   * MRStage1: preprocess the data 
   * */
  public static class MapStage1 extends MapReduceBase implements
          Mapper<LongWritable, Text, Text, Text> {

    private Text label = new Text(), text = new Text();

    private HashSet<String> htagmap;
    private HashMap<String, Integer> lbmap;

    private String htagPath = "";
    private String lbPath = "";

    public void configure(JobConf conf) {
      htagPath = conf.get("htagPath");
      lbPath = conf.get("lbPath");
      try {
        String vocaCacheName = new Path(htagPath).getName();
        String lbCacheName = new Path(lbPath).getName();
        Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);
        if (null != cacheFiles && cacheFiles.length > 0) {
          for (Path cachePath : cacheFiles) {
            if (cachePath.getName().equals(vocaCacheName)) {
              // System.out.println("cachePath: " + cachePath.toString());
              loadHashtag(cachePath);
            }
            else if(cachePath.getName().equals(lbCacheName)) {
              loadLabel(cachePath);
            }
          }
        }
      } catch (IOException ioe) {
        System.err.println("IOException reading from distributed cache");
        System.err.println(ioe.toString());
      }

    }
    
    private void loadLabel(Path lbPath) throws IOException {
      // note use of regular java.io methods here - this is a local file now
      BufferedReader br = new BufferedReader(new FileReader(lbPath.toString()));
      try {
        String line;
        lbmap = new HashMap<String, Integer>();
        while ((line = br.readLine()) != null) {
          String[] strs = line.split("\t");
          lbmap.put(strs[1], Integer.parseInt(strs[0]));
          // omit time series
          line = br.readLine();
          // System.out.println("neededWords READ: " + line);
        }
        // System.out.println("neededWords SIZE: " + neededWords.size());

      } finally {
        br.close();
      }
    }

    private void loadHashtag(Path htagPath) throws IOException {
      // note use of regular java.io methods here - this is a local file now
      BufferedReader br = new BufferedReader(new FileReader(htagPath.toString()));
      try {
        String line;
        htagmap = new HashSet<String>();
        while ((line = br.readLine()) != null) {
          htagmap.add(line);
          // System.out.println("neededWords READ: " + line);
        }
        // System.out.println("neededWords SIZE: " + neededWords.size());

      } finally {
        br.close();
      }
    }

    private Pattern pattern = Pattern.compile("#\\S+");

    public void map(LongWritable keyinit, Text value, final OutputCollector<Text, Text> output,
            final Reporter reporter) throws IOException {
      String line = value.toString();
      Matcher matcher = pattern.matcher(line);
      // Check all occurrence
      while (matcher.find()) {
        String cur = matcher.group();
        if (htagmap.contains(cur)) {
          //tag.set(cur);
//          String parsed = parseJson(line);
//          text.set(parsed);
          String jsonStr = line.substring(line.indexOf("\t")+1);
          String tagStr = line.substring(0, line.indexOf("\t"));
          int lb = 0;
          if(lbmap.containsKey(tagStr))
            lb = lbmap.get(tagStr);
          else 
            continue;
          // "tag \t json"
          text.set(tagStr + "\t" + jsonStr);
          label.set(lb + "");
          
          // group by label
          output.collect(label, text);
          break;
        }
      }

    }

    // return the useful section of json
    public String parseJson(String jsonLine) {
      JSONObject json = new JSONObject(jsonLine);

      String text = json.getString("text");
      String time = json.getString("created_at");

      return text + "\t" + time;
    }
  }

  public static class RedStage1 extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
    private Text word = new Text(), val = new Text();

    public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output,
            final Reporter reporter) throws IOException {
      while (values.hasNext()) {
        String curval = values.next().toString();
        val.set(curval);
        output.collect(key, val);
      }
    }

  }

  // Configure pass1
  protected JobConf configStage1() throws Exception {
    final JobConf conf = new JobConf(getConf(), PrepTSWithLabelGroupByLabel.class);
    conf.setJobName("PrepTSWithLabel");

    conf.set("htagPath", htagPath);
    conf.set("lbPath", lbPath);
    
    conf.setMapperClass(MapStage1.class);
    conf.setReducerClass(RedStage1.class);

    // first time finalout_path contains the initial vector
    FileInputFormat.setInputPaths(conf, inPath);
    FileOutputFormat.setOutputPath(conf, out1Path);

    // output gzip format
    conf.setBoolean("mapred.output.compress", true);
    conf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);

    conf.setNumReduceTasks(nreducers);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    // cache needed word
    DistributedCache.addCacheFile(new Path(htagPath).toUri(), conf);
    DistributedCache.addCacheFile(new Path(lbPath).toUri(), conf);
    
    return conf;
  }

  protected static int printUsage() {
    System.out.println("Usage: PrepTSWithLabel <inPath> <outPath> <htagPath> <lbPath> <# of reducers>");

    ToolRunner.printGenericCommandUsage(System.out);

    return -1;
  }

  public static void main(String[] args) throws Exception {
    final int result = ToolRunner.run(new Configuration(), new PrepTSWithLabelGroupByLabel(), args);

    System.exit(result);
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length != 5) {
      return printUsage();
    }

    inPath = new Path(args[0]);
    out1Path = new Path(args[1]);
    htagPath = args[2];
    lbPath = args[3];
    nreducers = Integer.parseInt(args[4]);

    // FileSystem fs = FileSystem.get(getConf());
    // String uri = getConf().get("fs.default.name");
    // FileSystem fs = FileSystem.get(URI.create(uri), getConf());
    // if (fs.exists(out1Path))
    // fs.delete(out1Path, true);
    // if (fs.exists(out2Path))
    // fs.delete(out2Path, true);

    System.out.println("\n-----===[Preprossing time series with label group by label]===-----\n");
    long startTime = System.currentTimeMillis();

    JobClient.runJob(configStage1());

    long endTime = System.currentTimeMillis();
    long elapse = (endTime - startTime) / 1000;

    System.out.println("\n[Preprossing time series with label group by label] Data Preprocessed.");
    System.out.println(String.format("[Preprossing time series with label group by label] total runing time: %d secs", elapse));

    return 0;
  }

}
