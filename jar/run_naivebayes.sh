hadoop dfs -rmr ./twitter
hadoop jar naivebayes.jar naivebayes.BackGroundWordCount ../zeyuanl/trend_serieslb/* ./twitter/bgwordcount 10
rm -rf output
mkdir output
hadoop fs -getmerge ./twitter/bgwordcount/* ./output/bgwordcount.txt
