# Generated automatically from Makefile.in by configure.
SWIG=swig
TARGET=TinySVM
JAVAC=
JAVA=java
JAR=
CXX=c++
LIBS=-ltinysvm
INC=

all:
	$(CXX) -O3 -c -fpic $(TARGET)_wrap.c  $(INC)
	$(CXX) -shared  $(TARGET)_wrap.o -o lib$(TARGET).so $(LIBS)
	$(JAVAC) *.java
	$(JAR) cfv $(TARGET).jar *.class

swig:
	swig -java -shadow -c++ $(TARGET).i

test:
	$(JAVA) -classpath $(TARGET).jar test

clean:
	rm -fr *.jar *.o *.so *.class   model*
	
cleanall:
	rm -fr BaseExample.java Example.java Model.java *.c
