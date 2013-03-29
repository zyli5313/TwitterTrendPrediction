# run whole data preprocessing pipeline

nreducer=3
#inpath='./trend_in/*((2009-0[6789(10)(11)(12)])|(2010-0[12]))-*'
#inpath='./trend_in/tweets.2009-06-01.gz'
inpath='./trend_in/*{2009-{06,07,08,09,10,11,12},2010-{01,02}}-*'
outpath='./trend_out/'
htagpath='./trend/topHTag.txt'

# put input file in HDFS
hadoop fs -rmr ./trend_out

# Usage: DataPreprocess <inPath> <outPath> <htagPath> <# of reducers>
hadoop jar prepro.jar prepro.DataPreprocess $inpath $outpath $htagpath $nreducer

rm ./trend.out
hadoop fs -getmerge ./trend_out ./trend.out
