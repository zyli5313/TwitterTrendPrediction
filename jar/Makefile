
.SUFFIXES: .java .class 
SOURCES = LRSgd.java 
CLASSES = LRSgd.class
JFLAGS = -g -classpath . -d .

mu = 0.1
R = 100000
T = 5

.java.class:
	javac $(JFLAGS) $*.java

default:
	javac $(JFLAGS) $(SOURCES)

all:
	javac $(JFLAGS) $(SOURCES)

demo:
	# Usage: LRSgd <trainFilePath> <testFilePath> <modelFilePath> <mu> <R/DicSize> <#iteration>
	java -cp trend.jar lr.LRSgd ../trend.train.gz ../trend.test.gz lr.model $(mu) $(R) $(T)

small:
	# Usage: LRSgd <trainFilePath> <testFilePath> <modelFilePath> <mu> <R/DicSize> <#iteration>
	java sgd.LRSgd data/trend.train.gz data/trend.test.gz data/lr.model $(mu) $(R) $(T)

clean:
	rm $(CLASSES)

doc: $(SOURCES)
	javadoc -author $(SOURCES)


