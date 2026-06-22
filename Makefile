DS_PATH ?= /opt/nvidia/deepstream/deepstream

CXX := g++
TARGET_LIB := custom_parser/libnvds_infercustom_yolov11_flat.so
SRCFILES := custom_parser/nvdsinfer_custom_yolov11_flat.cpp

CXXFLAGS := -Wall -std=c++17 -shared -fPIC
INCLUDES := -I$(DS_PATH)/sources/includes \
            -I/usr/local/cuda/include

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f $(TARGET_LIB)
