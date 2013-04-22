# run whole data preprocessing pipeline

nreducer=3
#inpath='./trend_in/*((2009-0[6789(10)(11)(12)])|(2010-0[12]))-*'
#inpath='./trend_in/tweets.2009-06-01.gz'
inpath='./trend_in/*{2009-{06,07,08,09,10,11,12},2010-{01,02}}-*'
outpath='./trend_out/'
outpath1=./trend_series/
outpath2=./trend_serieslb
htagpath='./trend/topHTag.txt'
lbpath=./trend/trendMerge.txt

# put input file in HDFS
hadoop fs -rmr ./trend_out

# Usage: DataPreprocess <inPath> <outPath> <htagPath> <# of reducers>
hadoop jar prepro.jar prepro.DataPreprocess $inpath $outpath $htagpath $nreducer

# Usage: PrepTimeSeries <inPath> <outPath> <# of reducers>
hadoop jar prepro.jar prepro.PrepTimeSeries $outpath $outpath1 $htagpath $nreducer

# Usage: PrepTSWithLabel <inPath> <outPath> <htagPath> <lbPath> <# of reducers>
hadoop jar prepro.jar prepro.PrepTSWithLabel $outpath $outpath2 $htagpath $lbpath $nreducer

rm ./trend.out
hadoop fs -getmerge ./trend_out ./trend.out
