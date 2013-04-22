package lr;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.json.JSONObject;

public class LRSgd {

  // private String fileTrainPath = "hw5/abstract.tiny.train";
  // private String fileTestPath = "hw5/abstract.tiny.test";
  // private String resPath = "hw5/tiny.res";
  private String fileTrainPath = "data/trend.train";
  private String fileTestPath = "data/trend.test";
  private String resPath = "data/res/res.txt";

  private String LCLPath = "data/lcl.txt";

  private double[][] W; // train 6 models

  private String modelPath = "data/lr.model";

  private static final int NUM_LBS = 6;

  private int R = 10000, T = 20;

  private static final double yita = 0.5, overflow = 20;

  private double mu = 0.1;

  private static final String[] lbs = { "1", "2", "3", "4", "5", "6" };

  private double lambda = 0.5;

  public LRSgd() {
    W = new double[NUM_LBS][R];
  }

  public LRSgd(String ftrain, String ftest, double mu, int R, int T) {
    fileTrainPath = ftrain;
    fileTestPath = ftest;
    this.mu = mu;
    this.R = R;
    this.T = T;
    W = new double[NUM_LBS][R];
  }

  protected double sigmoid(double score) {
    if (score > overflow)
      score = overflow;
    else if (score < -overflow)
      score = -overflow;
    double exp = Math.exp(-score);
    return 1 / (1 + exp);
  }

  public void train() {
    try {
      BufferedWriter bw = new BufferedWriter(new FileWriter(LCLPath));
      // for each iteration
      for (int t = 1; t <= T; t++) {
        lambda = yita / (t * t); // lambda decreases along iteration
        double totalLCL = 0.0;

        int k = 0;
        int[] A = new int[R];

        BufferedReader br = new BufferedReader(new FileReader(fileTrainPath));
        String line = null;

        while ((line = br.readLine()) != null) {
          // line: <label>\t<hashtag>\t<json>
          // construct V
          String[] strs = line.split("\t");
          HashSet<String> yset = new HashSet<String>();
          yset.add(strs[0]); // label set contains only one element

          JSONObject json = new JSONObject(strs[2]);

          // TODO: add tokenizer
          String text = json.getString("text");
          String[] words = text.split(" ");

          HashMap<Integer, Integer> V = new HashMap<Integer, Integer>();
          for (int j = 0; j < words.length; j++) {
            int h = words[j].hashCode() % R;
            if (h < 0)
              h += R;

            if (V.containsKey(h))
              V.put(h, V.get(h) + 1);
            else
              V.put(h, 1);
          }

          // calc Pik
          double[] Pi = new double[NUM_LBS];
          for (int yk = 0; yk < Pi.length; yk++) {
            double vw = 0.0;
            // sparse vector inner product
            for (Entry<Integer, Integer> ven : V.entrySet())
              vw += ven.getValue() * W[yk][ven.getKey()];
            // TODO: sigmoid?
            // Pi[yk] = 1.0 / (1 + Math.pow(Math.E, -vw));
            Pi[yk] = sigmoid(vw);
            // lcl
            int yik = yset.contains(lbs[yk]) ? 1 : 0;
            if (yik == 1)
              totalLCL += Math.log(Pi[yk]);
            else
              totalLCL += Math.log(1 - Pi[yk]);
          }

          k++;

          // update weight
          for (Entry<Integer, Integer> ven : V.entrySet()) {
            int h = ven.getKey();
            // order cannot change
            double reg = Math.pow(1 - 2 * lambda * mu, k - A[h]);

            // for each classifier
            for (int yk = 0; yk < NUM_LBS; yk++) {
              W[yk][h] *= reg;
              int yik = yset.contains(lbs[yk]) ? 1 : 0;
              W[yk][h] += lambda * (yik - Pi[yk]) * ven.getValue();
            }
            A[h] = k;
          }

        }

        // regularize remaining A[h]
        for (int h = 0; h < R; h++) {
          double reg = Math.pow(1 - 2 * lambda * mu, k - A[h]);
          for (int yk = 0; yk < NUM_LBS; yk++) {
            W[yk][h] *= reg;
          }
        }

        br.close();

        // output lcl
        System.out.println(String.format("iter: %d\t lcl: %f\n", t, totalLCL));
        bw.write(String.format("iter: %d\t lcl: %f\n", t, totalLCL));
      }

      // bw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
    System.out.println("Training Done!");
  }

  // public void test() {
  // BufferedReader br;
  // BufferedWriter bw;
  // try {
  // br = new BufferedReader(new FileReader(fileTestPath));
  // // bw = new BufferedWriter(new FileWriter(resPath));
  // String line = null;
  // long ntp = 0L, ntest = 0L;
  //
  // while ((line = br.readLine()) != null) {
  // ntest++;
  //
  // String[] strs = line.split("\t");
  // String[] ys = strs[0].split(",");
  // HashSet<String> gdlbs = new HashSet<String>();
  // for (String y : ys)
  // gdlbs.add(y);
  // String[] words = strs[1].split(" ");
  // int tabidx = line.indexOf("\t");
  //
  // HashMap<Integer, Integer> V = new HashMap<Integer, Integer>();
  // for (int j = 0; j < words.length; j++) {
  // int h = words[j].hashCode() % R;
  // if (h < 0)
  // h += R;
  //
  // if (V.containsKey(h))
  // V.put(h, V.get(h) + 1);
  // else
  // V.put(h, 1);
  // }
  //
  // // calc Pik
  // double[] Pi = new double[NUM_LBS];
  // String besty = "", bestyCorrect = "";
  // int ibesty = 0, ibestyCorrect = 0;
  // double pmax = 0.0, pmaxCorrect = 0.0;
  //
  // for (int yk = 0; yk < Pi.length; yk++) {
  // double vw = 0.0;
  // // sparse vector inner product
  // for (Entry<Integer, Integer> ven : V.entrySet())
  // vw += ven.getValue() * W[yk][ven.getKey()];
  // // TODO: sigmoid?
  // //Pi[yk] = 1.0 / (1 + Math.pow(Math.E, -vw));
  // Pi[yk] = sigmoid(vw);
  //
  // if (Pi[yk] > pmax) {
  // pmax = Pi[yk];
  // ibesty = yk;
  // besty = lbs[yk];
  // }
  // }
  //
  // if (gdlbs.contains(besty))
  // ntp++;
  //
  // String resline = String
  // .format("[%s]\t%s\t%f", line.substring(0, tabidx), besty, Pi[ibesty]);
  //
  // // bw.write(resline + "\n");
  // System.out.println(resline);
  // }
  //
  // String resline = String.format("Percent correct: %d/%d=%.1f%%", ntp, ntest, (double) ntp
  // / ntest * 100.0);
  // // bw.write(resline + "\n");
  // System.out.println(resline);
  //
  // // bw.close();
  // br.close();
  // } catch (FileNotFoundException e) {
  // // TODO Auto-generated catch block
  // e.printStackTrace();
  // } catch (IOException e) {
  // // TODO Auto-generated catch block
  // e.printStackTrace();
  // }
  //
  // }

  public void test() {
    BufferedReader br;
    BufferedWriter bw;
    try {
      br = new BufferedReader(new FileReader(fileTestPath));
      bw = new BufferedWriter(new FileWriter(resPath));
      String line = null;
      long ncorr = 0L, ntest = 0L;
      String lbpre = null, lb = null;
      ArrayList<String> words = new ArrayList<String>();

      while ((line = br.readLine()) != null) {
        // line: <hashtag> <lable> <json>
        String[] strs = line.split("\t");
        JSONObject json = new JSONObject(strs[2]);
        // TODO: add tokenizer
        String text = json.getString("text");
        String[] tokens = text.split(" ");
        
        // same hashtag
        if (lbpre == null || lbpre == lb) {
          for(String tk : tokens)
            words.add(tk);
        } 
        // diff hashtag, test
        else {
          ntest += NUM_LBS; // 6 classification results

          String gdlb = lbpre;  // golden label
          
          HashMap<Integer, Integer> V = new HashMap<Integer, Integer>();
          for (int j = 0; j < words.size(); j++) {
            int h = words.get(j).hashCode() % R;
            if (h < 0)
              h += R;

            if (V.containsKey(h))
              V.put(h, V.get(h) + 1);
            else
              V.put(h, 1);
          }

          // calc Pik
          double[] Pi = new double[NUM_LBS];
          // ArrayList<String> predLbs = new ArrayList<String>();
          StringBuilder sb = new StringBuilder();
          sb.append("[" + gdlb + "]");

          for (int yk = 0; yk < Pi.length; yk++) {
            double vw = 0.0;
            // sparse vector inner product
            for (Entry<Integer, Integer> ven : V.entrySet())
              vw += ven.getValue() * W[yk][ven.getKey()];
            // Pi[yk] = 1.0 / (1 + Math.pow(Math.E, -vw));
            Pi[yk] = sigmoid(vw);
            sb.append(String.format("\t%s\t%f", lbs[yk], Pi[yk]));

            // true positive
            if (Pi[yk] > 0.5 && gdlb.equals(lbs[yk]))
              ncorr++;
            // true negative
            else if (Pi[yk] < 0.5 && !gdlb.equals(lbs[yk]))
              ncorr++;
          }

          bw.write(sb.toString() + "\n");
          System.out.println(sb.toString());
          
          // for next iteration
          lbpre = lb;
          words = new ArrayList<String>();
          for(String tk : tokens)
            words.add(tk);
        }
      }
      
      // last test instance
      if(lbpre == lb) {
        ntest += NUM_LBS; // 6 classification results

        String gdlb = lbpre;  // golden label
        
        HashMap<Integer, Integer> V = new HashMap<Integer, Integer>();
        for (int j = 0; j < words.size(); j++) {
          int h = words.get(j).hashCode() % R;
          if (h < 0)
            h += R;

          if (V.containsKey(h))
            V.put(h, V.get(h) + 1);
          else
            V.put(h, 1);
        }

        // calc Pik
        double[] Pi = new double[NUM_LBS];
        // ArrayList<String> predLbs = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        sb.append("[" + gdlb + "]");

        for (int yk = 0; yk < Pi.length; yk++) {
          double vw = 0.0;
          // sparse vector inner product
          for (Entry<Integer, Integer> ven : V.entrySet())
            vw += ven.getValue() * W[yk][ven.getKey()];
          // Pi[yk] = 1.0 / (1 + Math.pow(Math.E, -vw));
          Pi[yk] = sigmoid(vw);
          sb.append(String.format("\t%s\t%f", lbs[yk], Pi[yk]));

          // true positive
          if (Pi[yk] > 0.5 && gdlb.equals(lbs[yk]))
            ncorr++;
          // true negative
          else if (Pi[yk] < 0.5 && !gdlb.equals(lbs[yk]))
            ncorr++;
        }

        bw.write(sb.toString() + "\n");
        System.out.println(sb.toString());
      }

      String resline = String.format("Percent correct: %d/%d=%.1f%%", ncorr, ntest, (double) ncorr
              / ntest * 100.0);
      bw.write(resline + "\n");
      System.out.println(resline);

      bw.close();
      br.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

  }

  public void saveLRModel() {
    try {
      BufferedWriter bw = new BufferedWriter(new FileWriter(modelPath));
      for (int i = 0; i < NUM_LBS; i++) {
        for (int j = 0; j < R; j++) {
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

  public static void main(String[] args) {
    // TODO Auto-generated method stub

    LRSgd lr = new LRSgd();
    lr.train();
    // lr.saveLRModel();
    lr.test();
  }
  /**
   * @param args
   */
  // public static void main(String[] args) {
  // // TODO Auto-generated method stub
  // if (args.length != 5) {
  // System.out.println("Usage: LRSgd <trainFilePath> <testFilePath> <mu> <R/DicSize> <#iteration>");
  // return;
  // }
  //
  // LRSgd lr = new LRSgd(args[0], args[1], Double.parseDouble(args[2]), Integer.parseInt(args[3]),
  // Integer.parseInt(args[4]));
  // lr.train();
  // // lr.saveLRModel();
  // lr.test();
  // }

}
