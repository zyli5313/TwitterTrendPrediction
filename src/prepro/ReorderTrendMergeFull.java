package prepro;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class ReorderTrendMergeFull {
  
  private final String trendPath = "data/trend.txt";
  private final String trendFullPath = "data/trend.seriesFull";
  private final String trendFullOutPath = "data/trend.seriesFull.order";
  
  /**
   * The order of "trendMerge.txt" and "trendMergeFull.txt" are not same. Reorder Full
   * @throws IOException 
   * */
  public void reorder() throws IOException {
    BufferedReader br = new BufferedReader(new FileReader(trendPath));
    BufferedReader brfull = new BufferedReader(new FileReader(trendFullPath));
    BufferedWriter bwfull = new BufferedWriter(new FileWriter(trendFullOutPath));
    HashMap<String, String> mapFull = new HashMap<String, String>();
    String line;
    
    while((line = brfull.readLine()) != null ){
      String next = brfull.readLine();
      mapFull.put(line.split("\\s+")[0], line + "\n" + next);
    }
    
    while((line = br.readLine()) != null) {
      String next = br.readLine();
      String key = line.split("\\s+")[0];
      
      bwfull.write(mapFull.get(key) + "\n");
    }
    
    br.close();
    brfull.close();
    
    System.out.println("Done");
  }

  /**
   * @param args
   * @throws IOException 
   */
  public static void main(String[] args) throws IOException {
    // TODO Auto-generated method stub
    ReorderTrendMergeFull reorder = new ReorderTrendMergeFull();
    reorder.reorder();
  }

}
