# extract hashtag from time series hashtag data

flb = file('../label.txt', 'r')
f = file('../trend.seriesFull', 'r')
fout = file('../trendMergeFull.txt', 'w')

while True:
    line = f.readline()
    if len(line) == 0:
        break
    series = f.readline()
    lb = flb.readline()
    
    fout.write(lb.rstrip('\n') + '\t' + line)
    fout.write(series)

f.close()
flb.close()
fout.close()

	
