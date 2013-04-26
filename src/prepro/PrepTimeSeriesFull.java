package prepro;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
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
import org.json.JSONException;
import org.json.JSONObject;

public class PrepTimeSeriesFull extends Configured implements Tool {

  private Path inPath, out1Path;

  public String htagPath;

  private int nreducers = 3;

  /**
   * MRStage1: preprocess the data 
   * */
  public static class MapStage1 extends MapReduceBase implements
          Mapper<LongWritable, Text, Text, Text> {

    private Text tag = new Text(), text = new Text();

    private HashSet<String> htagmap;

    private String htagPath = "";

    public void configure(JobConf conf) {
      htagPath = conf.get("htagPath");
      try {
        String vocaCacheName = new Path(htagPath).getName();
        Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);
        if (null != cacheFiles && cacheFiles.length > 0) {
          for (Path cachePath : cacheFiles) {
            if (cachePath.getName().equals(vocaCacheName)) {
              // System.out.println("cachePath: " + cachePath.toString());
              loadHashtag(cachePath);
              break;
            }
          }
        }
      } catch (IOException ioe) {
        System.err.println("IOException reading from distributed cache");
        System.err.println(ioe.toString());
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
          tag.set(cur);
          // "date \t 1"  date+counter
          String jsonStr = line.substring(line.indexOf("\t")+1);
          //System.out.println(jsonStr);
          
          String parsed = parseJson(jsonStr);
          // bad line
          if(parsed == null)
            continue;
          
          text.set(parsed);
          
          output.collect(tag, text);
          break;
        }
      }

    }

    // return the useful section of json
    public String parseJson(String jsonLine) {
      String outDate = "";
      try {
        JSONObject json = new JSONObject(jsonLine);
  
        String text = json.getString("text");
        String time = json.getString("created_at");
        // parse Twitter date
        SimpleDateFormat dateFormat = new SimpleDateFormat(
            "EEE MMM dd HH:mm:ss ZZZZZ yyyy", Locale.ENGLISH);
        dateFormat.setLenient(false);
        DateFormat dateFormatOut = new SimpleDateFormat("yyyyMMddHH");
        Date created;
        try {
          created = dateFormat.parse(time);
        } catch (Exception e) {
          return null;
        }
        outDate = dateFormatOut.format(created);
      }
      catch(JSONException je) {
        // bad line
        System.err.println(jsonLine);
        return null;
      }
      // cnt == 1
      return outDate + "\t" + "1";
    }
  }

  public static class RedStage1 extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
    private static final int IDX_PEAK = 128 / 3;
    private static final int NUM_TIME_SERIES = 128;
    
    private Text word = new Text(), val = new Text();
    private Map<String, Integer> map = new TreeMap<String, Integer>();
    
    
    public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output,
            final Reporter reporter) throws IOException {
      String keymax = "";
      int cntmax = 0;
      
      // build histogram based on time
      while (values.hasNext()) {
        // date + cnt
        String line = values.next().toString();
        String[] strs = line.split("\t");
        String date = strs[0];
        int cnt = Integer.parseInt(strs[1]);
        
        if(map.containsKey(date))
          map.put(date, map.get(date) + 1);
        else
          map.put(date, cnt);
        
        if(map.get(date) > cntmax) {
          keymax = date;
          cntmax = map.get(date);
        }
      }
      
      String[] keys = map.keySet().toArray(new String[map.size()]);
      StringBuilder sb = new StringBuilder();
      // 1stTime + \t + peakTime
      sb.append(keys[0] + "\t" + keymax + "\n");
      
      // TODO data points is not alligned with date time
      for(int i = 0; i < keys.length; i++) {
        sb.append(map.get(keys[i]) + "\t");
      }
      
      val.set(sb.toString());
      // "hashtag \t 1stTime + \t + peakTime + 
      // val(0) val(1) ... " (vaiable size. roughly 5760)
      output.collect(key, val);
    }

  }

  // Configure pass1
  protected JobConf configStage1() throws Exception {
    final JobConf conf = new JobConf(getConf(), PrepTimeSeriesFull.class);
    conf.setJobName("DataPreprocess");

    conf.set("htagPath", htagPath);
    
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
    
    return conf;
  }

  protected static int printUsage() {
    System.out.println("Usage: PrepTimeSeries <inPath> <outPath> <# of reducers>");

    ToolRunner.printGenericCommandUsage(System.out);

    return -1;
  }

  public static void main(String[] args) throws Exception {
    final int result = ToolRunner.run(new Configuration(), new PrepTimeSeriesFull(), args);

    System.exit(result);
  }

  @Override
  public int run(String[] args) throws Exception {
    if (args.length != 4) {
      return printUsage();
    }

    inPath = new Path(args[0]);
    out1Path = new Path(args[1]);
    htagPath = args[2];
    nreducers = Integer.parseInt(args[3]);

    // FileSystem fs = FileSystem.get(getConf());
    // String uri = getConf().get("fs.default.name");
    // FileSystem fs = FileSystem.get(URI.create(uri), getConf());
    // if (fs.exists(out1Path))
    // fs.delete(out1Path, true);
    // if (fs.exists(out2Path))
    // fs.delete(out2Path, true);

    System.out.println("\n-----===[Preprocessing time series]===-----\n");
    long startTime = System.currentTimeMillis();

    JobClient.runJob(configStage1());

    long endTime = System.currentTimeMillis();
    long elapse = (endTime - startTime) / 1000;

    System.out.println("\n[Preprocessing time series] Data Preprocessed.");
    System.out.println(String.format("[Preprocessing time series] total runing time: %d secs", elapse));

    return 0;
  }

}
