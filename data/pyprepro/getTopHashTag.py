# extract hashtag from time series hashtag data

f = file('../TwtHtag.txt', 'r')
fout = file('../topHTag.txt', 'w')

while True:
    line = f.readline()
    if len(line) == 0:
        break

    if '#' in line:
        strs = line.split('\t')
        for s in strs:
            if s.startswith('#'):
                fout.write(s + '\n');
f.close()
fout.close()

	
