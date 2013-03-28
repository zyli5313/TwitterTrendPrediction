import os
from urllib2 import urlopen, URLError, HTTPError
import urllib2

def dlfile(url):
    # Open the url
    try:
        f = urlopen(url)
        print "downloading " + url

        # Open our local file for writing
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())

    #handle errors
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url


def main():
    response = urllib2.urlopen('http://www.ark.cs.cmu.edu/tweets/files.daily.100.txt')
    html = response.read()
    infos = html.split('\n');
    preString = 'http://www.ark.cs.cmu.edu/tweets/'
    for info in infos:
        if info == '':
	    break 
        url = info.split('\t')
        allUrl = preString + url[0]
        dlfile(allUrl)

if __name__ == '__main__':
    main()

