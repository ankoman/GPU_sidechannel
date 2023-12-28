CXX=g++
NVCC=/usr/local/cuda-11.8/bin/nvcc
INCLUDE_DIR=/usr/include/hdf5/serial/
LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
LIBS=-lhdf5 -lhdf5_cpp
CXXFLAGS=-g
NVCCFLAGS=-g -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) $(LIBS)
#LD_LIBRARY_PATH=$(shell printenc LD_LIBRARY_PATH):/usr/local/cuda-11.8/lib64/
SRCDIR=src
SRCS=$(wildcard $(SRCDIR)/*.cu)
OBJDIR=obj
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))
TARGET=test

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)




# #-------------------------------------------------------------------------------
# # マクロ
# #====ここから===================================================================
# # オブジェクトファイル（ソースコード*.cu,*.c,*.cppを*.oに書き換えたもの）
# OBJS = hello.o
# # 実行ファイル名
# BIN = hello
# # CUDA環境のインストールパス
# CUDA_PATH = /usr/local/cuda/
# # デバイス認識用スクリプトのパス（CUI環境では必須，GUI環境ならばコメントアウト）
# RECOGINIZE_SCRIPT = /home/miettal/Dropbox/bin/recoginize.sh
# # OSの指定 LinuxかMacかを書く
# OS = Mac
# # 何ビットCPUか指定 32か64かを書く Macの場合は書かなくて良い
# CPU_BIT = 64
# #====ここまでを書き換える====================================================

# # CUDAコンパイラ
# NVCC = $(CUDA_PATH)/bin/nvcc
# # CUDAインクルードディレクトリ
# CUDA_INCLUDE_DIR = $(CUDA_PATH)/include/
# # CUDAライブラリディレクトリ（32bitの場合はlib，64bitの場合はlib64）
# # 手動でしたい場合はコメントを外して，記述する．
# #CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib/
# # C++コンパイラ
# #（C++対応GCC 4.4を指定してください．そうしないとリンクでエラーが出ます．）
# #  インストール方法 例 Debian系 $sudo apt-get install gcc-4.4 g++-4.4
# #                   例 mac XCODEをインストールする
# # 手動でしたい場合はコメントを外して，記述する．
# #CXX = gcc-4.4
# ifeq ($(OS), Linux)
#   ifndef CXX
#     CXX = gcc-4.4
#   endif
#   ifndef CUDA_LIBRARY_DIR
#     ifeq ($(CPU_BIT), 32)
#       CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib/
#     else
#       CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib64/
#     endif
#   endif
# else
#   ifeq ($(OS), Mac)
#     ifndef CXX
#       CXX = gcc
#     endif
#     ifndef CUDA_LIBRARY_DIR
#       CUDA_LIBRARY_DIR = $(CUDA_PATH)/lib/
#     endif
#   endif
# endif

# # コンパイルオプション
# CFLAGS = -I$(CUDA_INCLUDE_DIR)
# # リンクオプション
# LDFLAGS = -L$(CUDA_LIBRARY_DIR) -lcudart 

# #-------------------------------------------------------------------------------
# # 生成ルール
# # コンパイル
# all:$(BIN)

# # 実行
# run:$(BIN)
#   $(RECOGINIZE_SCRIPT)
# 	./$(BIN)

# # 削除
# clean:
# 	rm $(OBJS)
# 	rm $(BIN)

# # オブジェクトファイルをリンクし，実行ファイルを作成する．
# $(BIN):$(OBJS)
# 	$(CXX) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $(BIN)

# #-------------------------------------------------------------------------------
# # サフィックスルール
# # サフィックス
# .SUFFIXES: .o .cpp .cu

# # cuファイルをnvccでcppファイルに変換
# .cu.cpp:
# 	$(NVCC) --cuda $< -o $@

# # cppファイルをコンパイルし，オブジェクトファイルを作成する．
# .cpp.o:
# 	$(CXX) $(CFLAGS) -c $<

# # cファイルをコンパイルし，オブジェクトファイルを作成する．
# .c.o:
# 	$(CXX) $(CFLAGS) -c $<