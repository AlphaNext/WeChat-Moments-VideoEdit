CXX = g++
CXXFLAGS := `pkg-config --cflags opencv` `pkg-config --cflags magick++` 
LIBS := `pkg-config --libs opencv` `pkg-config --libs magick++` 

OBJFILES += TrackGIF.o
SOURCE = TrackGIF.cpp

TARGET = main 
CFLAGS += -c -g -Wall -std=c++11 
all:$(TARGET)

$(OBJFILES):$(SOURCE)
	$(CXX) $(CFLAGS) $(SOURCE) $(CXXFLAGS)

$(TARGET):$(OBJFILES)
	$(CXX) -o $(TARGET) $(OBJFILES) -I $(CXXFLAGS) $(LIBS) 
clean:
	rm -rf $(OBJFILES) $(TARGET)

