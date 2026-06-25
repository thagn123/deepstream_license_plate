CUDA_VER?=13.1
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

# NVIDIA DeepStream version (typically 6.3 or 6.4 for newer versions, adjust if needed)
DS_VER=6.4

CXX:=g++
NVCC:=/usr/local/cuda/bin/nvcc

SRCS:= gstlaplacian.cpp laplacian_lib.cu
INCS:= $(wildcard *.h)

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

GST_INSTALL_DIR?=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/
LIB_INSTALL_DIR?=/usr/lib/x86_64-linux-gnu/

CFLAGS+= -fPIC -DDS_VERSION=\"6.4.0\" \
	 -I /usr/local/cuda/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes

GST_CFLAGS:= $(shell pkg-config --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4)
LIBS:= $(shell pkg-config --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4)

LIBS+= -L/usr/local/cuda/lib64/ -lcudart \
       -L/opt/nvidia/deepstream/deepstream/lib/ \
       -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib/ \
       -lnvdsgst_meta -lnvbufsurface -lnvbufsurftransform -lnvds_meta

TARGET_LIB:= libnvdsgst_laplacian.so

all: $(TARGET_LIB)

%.o: %.cpp $(INCS)
	$(CXX) -c -o $@ $(CFLAGS) $(GST_CFLAGS) $<

%.o: %.cu $(INCS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB): $(OBJS)
	$(CXX) -shared -o $@ $^ $(CFLAGS) $(GST_CFLAGS) $(LIBS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
