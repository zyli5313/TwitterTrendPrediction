package lr;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.zip.GZIPInputStream;

import org.json.JSONException;
import org.json.JSONObject;

import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.LowerCaseTokenizerFactory;
import com.aliasi.tokenizer.PorterStemmerTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;

public class LRSgdTime {
  
  // class for time series
  class TS {
    public static final int VAL_SIZE = 128;
    
    public String lb; // lable
    public String ht; //hashtag
    public String startTime;
    public String peakTime;
    public double[] vals; // time series
    
    public TS(String l1, String l2) {
      String[] strs = l1.split("\\s+");
      lb = strs[0];
      ht = strs[1];
      startTime = strs[2];
      peakTime = strs[3];
      
      strs = l2.split("\\s+");
      ArrayList<Double> vallist = new ArrayList<Double>();
      for(int i = 0; i < strs.length; i++)
        vallist.add(Double.parseDouble(strs[i]));
      
      vals = new double[vallist.size()];
      for(int i = 0; i < vallist.size(); i++)
        vals[i] = vallist.get(i);
    }
  }
  

  private String fileTrainPath = "data/trendMerge.txt";
  private String fileTestPath = "data/trendMergeFull.txt";
//  private String fileTestPath = "data/trendMerge.txt";
  private List<TS> tsTrain; // time series for train
  private List<TS> tsTest;  // time series for test

  private String resPath = "data/res/res.txt";

  private String LCLPath = "data/lcl.txt";

  private double[][] W, Wold; // train 6 models

  private String modelPath = "data/lrtime.model";

  private static final int NUM_LBS = 6;

  private int interval = 128, T = 10;

  private static final double yita = 0.5, overflow = 20;

  private double mu = 0.1;

  private static final String[] lbs = { "1", "2", "3", "4", "5", "6" };

  private double lambda = 0.5;
  
  private TokenizerFactory tokFactory;

  public LRSgdTime() {
    W = new double[NUM_LBS][interval];
    Wold = new double[NUM_LBS][interval];
    
    tokFactory = new NormalizedTokenizerFactory();
    tokFactory = new LowerCaseTokenizerFactory(tokFactory);
    tokFactory = new EnglishStopTokenizerFactory(tokFactory);
    tokFactory = new PorterStemmerTokenizerFactory(tokFactory);
    
    tsTrain = new ArrayList<TS>();
    tsTest = new ArrayList<TS>();
    // read in the train and test file
    try {
      readTrainTest();
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(-1);
    }
  }
  
  public LRSgdTime(String ftrain, String ftest, double mu, int R, int T) {
    this();
    fileTrainPath = ftrain;
    fileTestPath = ftest;
    this.mu = mu;
    this.interval = R;
    this.T = T;
  }
  
  private void readTrainTest() throws IOException {
    int trainNum = 798; // train:test==8:2 (total 998)
    
    // read train
    BufferedReader br = new BufferedReader(new FileReader(fileTrainPath));;
    try {
      String line;
      int cnt = 0;
      while((line = br.readLine()) != null) {
        String lineNext = br.readLine();
        TS series = new TS(line, lineNext);
        cnt++;
        
        if(cnt < trainNum)
          tsTrain.add(series);
        else
          break;
      }
    }
    finally {
      br.close();
    }
    
    // read test
    br = new BufferedReader(new FileReader(fileTestPath));;
    try {
      String line;
      int cnt = 0;
      while((line = br.readLine()) != null) {
        String lineNext = br.readLine();
        TS series = new TS(line, lineNext);
        cnt++;
        
        if(cnt >= trainNum)
          tsTest.add(series);
      }
    }
    finally {
      br.close();
    }
  }

  protected double sigmoid(double score) {
    if (score > overflow)
      score = overflow;
    else if (score < -overflow)
      score = -overflow;
    double exp = Math.exp(-score);
    return 1 / (1 + exp);
  }
  
  /**
   * 
   * @return jump point start idx
   * */
  private int detectJump(TS ts) {
    final int times = 3;
    final int upSize = 100;
    final double jumpRate = 0.904238522425;  // is jump point if > this jumpRate (calc from 6 canonical trends of labels)
    
    int s = 0, e = s + interval;
    double mean = 0.0, meanPre = 0.0, var = 0.0;
    //double lastVal = ts.vals[s + interval];
    double jumpness = 0.0;
    
    while(e < ts.vals.length && (jumpness < jumpRate || mean < (times * meanPre)) ) {
      // a potential jump point
      if(Math.abs(ts.vals[s] - ts.vals[e]) > upSize) {
        mean = getMean(ts.vals, s, e);
        var = getVar(ts.vals, s, e, mean);
        meanPre = getMean(ts.vals, 0, s);
        
        jumpness = var / mean;
      }
      
      s++;
      e = s + interval;
    }
    
    return s;
  }
  
  private double getMean(double[] vals, int s, int e) {
    double sum = 0.0;
    for(int i = s; i < e; i++)
      sum += vals[i];
    
    return sum / (e-s+1);
  }
  
  private double getVar(double[] vals, int s, int e, double mean) {
    double var = 0.0;
    for(int i = s; i < e; i++)
      var += (vals[i]-mean) * (vals[i]-mean);
    
    return Math.sqrt(var);
  }

  public void train() {
    int badcnt = 0;
    try {
      BufferedWriter bw = new BufferedWriter(new FileWriter(LCLPath));
      
      // for each iteration
      for (int t = 1; t <= T; t++) {
        lambda = yita / (t * t); // lambda decreases along iteration

        int k = 0;
        int[] A = new int[interval];

        for(int i = 0; i < tsTrain.size(); i++) {
          // construct V
          TS ts = tsTrain.get(i);
          HashSet<String> yset = new HashSet<String>();
          yset.add(ts.lb); // label set contains only one element

//          HashMap<Integer, Integer> V = new HashMap<Integer, Integer>();
//          for (int j = 0; j < words.length; j++) {
//            int h = words[j].hashCode() % R;
//            if (h < 0)
//              h += R;
//
//            if (V.containsKey(h))
//              V.put(h, V.get(h) + 1);
//            else
//              V.put(h, 1);
//          }
          
          // calc Pik
          double[] Pi = new double[NUM_LBS];
          for (int yk = 0; yk < Pi.length; yk++) {
            double vw = 0.0;
            
            // train limit to size R (use first R element to train)
            for(int j = 0; j < interval; j++)
              vw += ts.vals[j] * W[yk][j];
            Pi[yk] = sigmoid(vw);
          }

          k++;

          // update weight
          for (int j = 0; j < interval; j++) {
            if(Math.abs(ts.vals[j] - 0.0) < 1e-3)
              continue;
            
            // order cannot change
            double reg = Math.pow(1 - 2 * lambda * mu, k - A[j]);

            // for each classifier
            for (int yk = 0; yk < NUM_LBS; yk++) {
              W[yk][j] *= reg;
              int yik = yset.contains(lbs[yk]) ? 1 : 0;
              W[yk][j] += lambda * (yik - Pi[yk]) * ts.vals[j];
            }
            A[j] = k;
          }

        }

        // regularize remaining A[h]
        for (int h = 0; h < interval; h++) {
          double reg = Math.pow(1 - 2 * lambda * mu, k - A[h]);
          for (int yk = 0; yk < NUM_LBS; yk++) {
            W[yk][h] *= reg;
          }
        }

        // check if W is converging
        double diff = 0.0, sum = 0.0;
        for (int i = 0; i < W.length; i++)
          for (int j = 0; j < W[0].length; j++) {
            diff += Math.abs(W[i][j] - Wold[i][j]);
            sum += W[i][j] + Wold[i][j];
            Wold[i][j] = W[i][j];
          }

        diff /= W.length * W[0].length;
        //System.out.println(String.format("it:%d\tw diff:%f\tsum:%f", t, diff, sum));

        // output progress
        // System.out.print(".");
      }

      // bw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    //System.out.println("Training Done! bad cnt:" + badcnt);
  }

  public void test() {
    BufferedWriter bw;
    try {
      // br = new BufferedReader(new FileReader(fileTestPath));
      bw = new BufferedWriter(new FileWriter(resPath));
      int[] ntp = new int[NUM_LBS]; // # true positive
      int[] nfn = new int[NUM_LBS];
      int[] nfp = new int[NUM_LBS];
      int ntest = 0, ncorr = 0;

      for(int i = 0; i < tsTest.size(); i++) {
        TS ts = tsTest.get(i);
        // System.out.println(++cnt + "\tlbpre:" + lbpre + "\tlb:" + lb);

        ntest++; // 1 classification results
        String gdlb = ts.lb; // golden label
        
        // test on the first most salient jump point
        // TODO: detect jump point
        if(ts.vals.length < interval)
          continue;
        int startIdx = detectJump(ts);  // jump start point
//        int startIdx = 0;
        
        // calc Pik
        double[] Pi = new double[NUM_LBS];
        // ArrayList<String> predLbs = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        sb.append("[" + gdlb + "]");

        double pmax = 0.0;
        String lbmax = "";
        for (int yk = 0; yk < Pi.length; yk++) {
          double vw = 0.0;
          
          // test limit to size R (use first R element to test)
          for(int j = 0; j < interval; j++)
            vw += ts.vals[startIdx + j] * W[yk][j];
          Pi[yk] = sigmoid(vw);
          sb.append(String.format("\t%s\t%e", lbs[yk], Pi[yk]));

          if (Pi[yk] > pmax) {
            pmax = Pi[yk];
            lbmax = lbs[yk];
          }
          
          // evaluate per classifier performance
          if(gdlb.equals(lbs[yk]) && Pi[yk] > 0.5)
            ntp[yk]++;
          if(!gdlb.equals(lbs[yk]) && Pi[yk] > 0.5)
            nfp[yk]++;
          if(gdlb.equals(lbs[yk]) && Pi[yk] < 0.5)
            nfn[yk]++;
        }

        if (gdlb.equals(lbmax))
          ncorr++;

//        bw.write("pred:" + lbmax + "\t" + sb.toString() + "\n");
//        System.out.println("pred:" + lbmax + "\t" + sb.toString());

      }
      
      // micro-avarge over all testing instances
      double precision = 0.0, recall = 0.0, fscore = 0.0, 
              sumtp = 0.0, sumfn = 0.0, sumfp = 0.0;
      for(int k = 0; k < NUM_LBS; k++) {
        sumtp += ntp[k];
        sumfn += nfn[k];
        sumfp += nfp[k];
      }
      precision = sumtp / (sumtp + sumfp);
      recall = sumtp / (sumtp + sumfn);
      fscore = 2*precision*recall / (precision + recall);
   
//      String resline = String.format("Percent correct: %d/%d=%.1f%%", ntp, ntest, (double) ncorr
//              / ntest * 100.0);
      String resline = String.format("P:%.2f\tR:%.2f\tF:%.2f", precision, recall, fscore);
      bw.write(resline + "\n");
      System.out.println(resline);

      bw.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

  }
  
  public void saveLRModel() {
    saveLRModel(modelPath);
  }

  public void saveLRModel(String mpath) {
    modelPath = mpath;
    try {
      BufferedWriter bw = new BufferedWriter(new FileWriter(modelPath));
      for (int i = 0; i < NUM_LBS; i++) {
        for (int j = 0; j < interval; j++) {
          bw.write(W[i][j] + " ");
        }
        bw.write("\n");
      }

      bw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }

    // System.out.println("Save model Done!");
  }

  public void loadLRModel() {
    try {
      BufferedReader br = new BufferedReader(new FileReader(modelPath));
      String line = null;
      int i = 0;
      while ((line = br.readLine()) != null) {
        String[] strs = line.split(" ");
        for (int j = 0; j < strs.length; j++)
          W[i][j] = Double.parseDouble(strs[j]);
        i++;
      }

      br.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    // System.out.println("Load model Done!");
  }

  public String[] tokenizeDoc(String tweet) {
    List<String> tokens = new ArrayList<String>();
    
    char[] chars = tweet.toCharArray();
    Tokenizer tokenizer = tokFactory.tokenizer(chars, 0, chars.length);
    String token;
    while ((token = tokenizer.nextToken()) != null) {
      token = token.toLowerCase();
      tokens.add(token);
    }
    
    String[] words = new String[tokens.size()];
    for(int i = 0; i < tokens.size(); i++)
      words[i] = tokens.get(i);
    return words;
  }

   public static void main(String[] args) {
     LRSgdTime lr = new LRSgdTime();
     for(int i = 1; i < 129; i++) {
       System.out.print("itv: " + i + "\t");
       lr.interval = i;
       lr.train();
       //lr.saveLRModel();
       lr.test();
     }
   }
   
//   public static void main(String[] args) {
//     LRSgdTime lr = new LRSgdTime();
//     lr.train();
//     lr.saveLRModel();
//     lr.test();
//   }

  /**
   * @param args
   */
//  public static void main(String[] args) {
//    // TODO Auto-generated method stub
//    if (args.length != 6) {
//      System.out
//              .println("Usage: LRSgd <trainPath> <testPath> <modelFilePath> <mu> <R/interval> <#iteration>");
//      return;
//    }
//
//    LRSgdTime lr = new LRSgdTime(args[0], args[1], Double.parseDouble(args[3]), Integer.parseInt(args[4]),
//            Integer.parseInt(args[5]));
//    lr.train();
//    lr.saveLRModel(args[2]);
//    lr.test();
//  }

}
