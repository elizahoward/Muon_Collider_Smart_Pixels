#include "hls_signal_handler.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <vector>
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_directio.h"
#include "hls_stream.h"
using namespace std;

namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}


namespace hls::sim
{
  struct Buffer {
    char *first;
    Buffer(char *addr) : first(addr)
    {
    }
  };

  struct DBuffer : public Buffer {
    static const size_t total = 1<<10;
    size_t ufree;

    DBuffer(size_t usize) : Buffer(nullptr), ufree(total)
    {
      first = new char[usize*ufree];
    }

    ~DBuffer()
    {
      delete[] first;
    }
  };

  struct CStream {
    char *front;
    char *back;
    size_t num;
    size_t usize;
    std::list<Buffer*> bufs;
    bool dynamic;

    CStream() : front(nullptr), back(nullptr),
                num(0), usize(0), dynamic(true)
    {
    }

    ~CStream()
    {
      for (Buffer *p : bufs) {
        delete p;
      }
    }

    template<typename T>
    T* data()
    {
      return (T*)front;
    }

    template<typename T>
    void transfer(hls::stream<T> *param)
    {
      while (!empty()) {
        param->write(*(T*)nextRead());
      }
    }

    bool empty();
    char* nextRead();
    char* nextWrite();
  };

  bool CStream::empty()
  {
    return num == 0;
  }

  char* CStream::nextRead()
  {
    assert(num > 0);
    char *res = front;
    front += usize;
    if (dynamic) {
      if (++static_cast<DBuffer*>(bufs.front())->ufree == DBuffer::total) {
        if (bufs.size() > 1) {
          bufs.pop_front();
          front = bufs.front()->first;
        } else {
          front = back = bufs.front()->first;
        }
      }
    }
    --num;
    return res;
  }

  char* CStream::nextWrite()
  {
    if (dynamic) {
      if (static_cast<DBuffer*>(bufs.back())->ufree == 0) {
        bufs.push_back(new DBuffer(usize));
        back = bufs.back()->first;
      }
      --static_cast<DBuffer*>(bufs.back())->ufree;
    }
    char *res = back;
    back += usize;
    ++num;
    return res;
  }

  std::list<CStream> streams;
  std::map<char*, CStream*> prebuilt;

  CStream* createStream(size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = true;
      s.bufs.push_back(new DBuffer(usize));
      s.front = s.bufs.back()->first;
      s.back = s.front;
      s.num = 0;
      s.usize = usize;
    }
    return &s;
  }

  template<typename T>
  CStream* createStream(hls::stream<T> *param)
  {
    CStream *s = createStream(sizeof(T));
    {
      s->dynamic = true;
      while (!param->empty()) {
        T data = param->read();
        memcpy(s->nextWrite(), (char*)&data, sizeof(T));
      }
      prebuilt[s->front] = s;
    }
    return s;
  }

  template<typename T>
  CStream* createStream(T *param, size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = false;
      s.bufs.push_back(new Buffer((char*)param));
      s.front = s.back = s.bufs.back()->first;
      s.usize = usize;
      s.num = ~0UL;
    }
    prebuilt[s.front] = &s;
    return &s;
  }

  CStream* findStream(char *buf)
  {
    return prebuilt.at(buf);
  }
}
class AESL_RUNTIME_BC {
  public:
    AESL_RUNTIME_BC(const char* name) {
      file_token.open( name);
      if (!file_token.good()) {
        cout << "Failed to open tv file " << name << endl;
        exit (1);
      }
      file_token >> mName;//[[[runtime]]]
    }
    ~AESL_RUNTIME_BC() {
      file_token.close();
    }
    int read_size () {
      int size = 0;
      file_token >> mName;//[[transaction]]
      file_token >> mName;//transaction number
      file_token >> mName;//pop_size
      size = atoi(mName.c_str());
      file_token >> mName;//[[/transaction]]
      return size;
    }
  public:
    fstream file_token;
    string mName;
};
using hls::sim::Byte;
struct __cosim_s546__ { char data[1024]; };
struct __cosim_s1024__ { char data[1024]; };
extern "C" void myproject(volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, Byte<2>*, volatile void *, volatile void *, Byte<2>*, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, Byte<2>*, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, Byte<2>*, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, volatile void *, Byte<2>*, volatile void *);
extern "C" void apatb_myproject_hw(volatile void * __xlx_apatb_param_cluster, volatile void * __xlx_apatb_param_nModule, volatile void * __xlx_apatb_param_x_local, volatile void * __xlx_apatb_param_y_local, volatile void * __xlx_apatb_param_layer29_out, volatile void * __xlx_apatb_param_w9, volatile void * __xlx_apatb_param_b9_0, volatile void * __xlx_apatb_param_b9_1, volatile void * __xlx_apatb_param_w16, volatile void * __xlx_apatb_param_b16_0, volatile void * __xlx_apatb_param_b16_1, volatile void * __xlx_apatb_param_b16_2, volatile void * __xlx_apatb_param_b16_3, volatile void * __xlx_apatb_param_b16_4, volatile void * __xlx_apatb_param_b16_5, volatile void * __xlx_apatb_param_b16_6, volatile void * __xlx_apatb_param_b16_7, volatile void * __xlx_apatb_param_b16_8, volatile void * __xlx_apatb_param_b16_9, volatile void * __xlx_apatb_param_b16_10, volatile void * __xlx_apatb_param_b16_11, volatile void * __xlx_apatb_param_b16_12, volatile void * __xlx_apatb_param_b16_13, volatile void * __xlx_apatb_param_b16_14, volatile void * __xlx_apatb_param_b16_15, volatile void * __xlx_apatb_param_w21, volatile void * __xlx_apatb_param_b21_0, volatile void * __xlx_apatb_param_b21_1, volatile void * __xlx_apatb_param_b21_2, volatile void * __xlx_apatb_param_b21_3, volatile void * __xlx_apatb_param_b21_4, volatile void * __xlx_apatb_param_b21_5, volatile void * __xlx_apatb_param_b21_6, volatile void * __xlx_apatb_param_b21_7, volatile void * __xlx_apatb_param_b21_8, volatile void * __xlx_apatb_param_b21_9, volatile void * __xlx_apatb_param_b21_10, volatile void * __xlx_apatb_param_b21_11, volatile void * __xlx_apatb_param_b21_12, volatile void * __xlx_apatb_param_b21_13, volatile void * __xlx_apatb_param_b21_14, volatile void * __xlx_apatb_param_b21_15, volatile void * __xlx_apatb_param_b21_16, volatile void * __xlx_apatb_param_b21_17, volatile void * __xlx_apatb_param_b21_18, volatile void * __xlx_apatb_param_b21_19, volatile void * __xlx_apatb_param_b21_20, volatile void * __xlx_apatb_param_b21_21, volatile void * __xlx_apatb_param_b21_22, volatile void * __xlx_apatb_param_b21_23, volatile void * __xlx_apatb_param_b21_24, volatile void * __xlx_apatb_param_b21_25, volatile void * __xlx_apatb_param_b21_26, volatile void * __xlx_apatb_param_b21_27, volatile void * __xlx_apatb_param_b21_28, volatile void * __xlx_apatb_param_b21_29, volatile void * __xlx_apatb_param_b21_30, volatile void * __xlx_apatb_param_b21_31, volatile void * __xlx_apatb_param_b21_32, volatile void * __xlx_apatb_param_b21_33, volatile void * __xlx_apatb_param_b21_34, volatile void * __xlx_apatb_param_b21_35, volatile void * __xlx_apatb_param_b21_36, volatile void * __xlx_apatb_param_b21_37, volatile void * __xlx_apatb_param_b21_38, volatile void * __xlx_apatb_param_b21_39, volatile void * __xlx_apatb_param_b21_40, volatile void * __xlx_apatb_param_b21_41, volatile void * __xlx_apatb_param_b21_42, volatile void * __xlx_apatb_param_b21_43, volatile void * __xlx_apatb_param_b21_44, volatile void * __xlx_apatb_param_b21_45, volatile void * __xlx_apatb_param_b21_46, volatile void * __xlx_apatb_param_b21_47, volatile void * __xlx_apatb_param_b21_48, volatile void * __xlx_apatb_param_b21_49, volatile void * __xlx_apatb_param_b21_50, volatile void * __xlx_apatb_param_b21_51, volatile void * __xlx_apatb_param_b21_52, volatile void * __xlx_apatb_param_b21_53, volatile void * __xlx_apatb_param_b21_54, volatile void * __xlx_apatb_param_b21_55, volatile void * __xlx_apatb_param_b21_56, volatile void * __xlx_apatb_param_b21_57, volatile void * __xlx_apatb_param_b21_58, volatile void * __xlx_apatb_param_b21_59, volatile void * __xlx_apatb_param_b21_60, volatile void * __xlx_apatb_param_b21_61, volatile void * __xlx_apatb_param_b21_62, volatile void * __xlx_apatb_param_b21_63, volatile void * __xlx_apatb_param_b21_64, volatile void * __xlx_apatb_param_b21_65, volatile void * __xlx_apatb_param_b21_66, volatile void * __xlx_apatb_param_b21_67, volatile void * __xlx_apatb_param_b21_68, volatile void * __xlx_apatb_param_b21_69, volatile void * __xlx_apatb_param_b21_70, volatile void * __xlx_apatb_param_b21_71, volatile void * __xlx_apatb_param_w24, volatile void * __xlx_apatb_param_b24_0, volatile void * __xlx_apatb_param_b24_1, volatile void * __xlx_apatb_param_b24_2, volatile void * __xlx_apatb_param_b24_3, volatile void * __xlx_apatb_param_b24_4, volatile void * __xlx_apatb_param_b24_5, volatile void * __xlx_apatb_param_b24_6, volatile void * __xlx_apatb_param_b24_7, volatile void * __xlx_apatb_param_b24_8, volatile void * __xlx_apatb_param_b24_9, volatile void * __xlx_apatb_param_b24_10, volatile void * __xlx_apatb_param_b24_11, volatile void * __xlx_apatb_param_b24_12, volatile void * __xlx_apatb_param_b24_13, volatile void * __xlx_apatb_param_b24_14, volatile void * __xlx_apatb_param_b24_15, volatile void * __xlx_apatb_param_b24_16, volatile void * __xlx_apatb_param_b24_17, volatile void * __xlx_apatb_param_b24_18, volatile void * __xlx_apatb_param_b24_19, volatile void * __xlx_apatb_param_b24_20, volatile void * __xlx_apatb_param_b24_21, volatile void * __xlx_apatb_param_b24_22, volatile void * __xlx_apatb_param_b24_23, volatile void * __xlx_apatb_param_b24_24, volatile void * __xlx_apatb_param_b24_25, volatile void * __xlx_apatb_param_b24_26, volatile void * __xlx_apatb_param_b24_27, volatile void * __xlx_apatb_param_b24_28, volatile void * __xlx_apatb_param_b24_29, volatile void * __xlx_apatb_param_b24_30, volatile void * __xlx_apatb_param_b24_31, volatile void * __xlx_apatb_param_b24_32, volatile void * __xlx_apatb_param_b24_33, volatile void * __xlx_apatb_param_b24_34, volatile void * __xlx_apatb_param_b24_35, volatile void * __xlx_apatb_param_b24_36, volatile void * __xlx_apatb_param_b24_37, volatile void * __xlx_apatb_param_b24_38, volatile void * __xlx_apatb_param_b24_39, volatile void * __xlx_apatb_param_b24_40, volatile void * __xlx_apatb_param_b24_41, volatile void * __xlx_apatb_param_b24_42, volatile void * __xlx_apatb_param_b24_43, volatile void * __xlx_apatb_param_b24_44, volatile void * __xlx_apatb_param_b24_45, volatile void * __xlx_apatb_param_b24_46, volatile void * __xlx_apatb_param_b24_47, volatile void * __xlx_apatb_param_b24_48, volatile void * __xlx_apatb_param_b24_49, volatile void * __xlx_apatb_param_b24_50, volatile void * __xlx_apatb_param_b24_51, volatile void * __xlx_apatb_param_b24_52, volatile void * __xlx_apatb_param_b24_53, volatile void * __xlx_apatb_param_b24_54, volatile void * __xlx_apatb_param_b24_55, volatile void * __xlx_apatb_param_b24_56, volatile void * __xlx_apatb_param_b24_57, volatile void * __xlx_apatb_param_w27, volatile void * __xlx_apatb_param_b27) {
using hls::sim::createStream;
  // Collect __xlx_w9__tmp_vec
std::vector<Byte<2>> __xlx_w9__tmp_vec;
for (size_t i = 0; i < 18; ++i){
__xlx_w9__tmp_vec.push_back(((Byte<2>*)__xlx_apatb_param_w9)[i]);
}
  int __xlx_size_param_w9 = 18;
  int __xlx_offset_param_w9 = 0;
  int __xlx_offset_byte_param_w9 = 0*2;
  // Collect __xlx_w16__tmp_vec
std::vector<Byte<2>> __xlx_w16__tmp_vec;
for (size_t i = 0; i < 48; ++i){
__xlx_w16__tmp_vec.push_back(((Byte<2>*)__xlx_apatb_param_w16)[i]);
}
  int __xlx_size_param_w16 = 48;
  int __xlx_offset_param_w16 = 0;
  int __xlx_offset_byte_param_w16 = 0*2;
  // Collect __xlx_w21__tmp_vec
std::vector<Byte<2>> __xlx_w21__tmp_vec;
for (size_t i = 0; i < 9792; ++i){
__xlx_w21__tmp_vec.push_back(((Byte<2>*)__xlx_apatb_param_w21)[i]);
}
  int __xlx_size_param_w21 = 9792;
  int __xlx_offset_param_w21 = 0;
  int __xlx_offset_byte_param_w21 = 0*2;
  // Collect __xlx_w24__tmp_vec
std::vector<Byte<2>> __xlx_w24__tmp_vec;
for (size_t i = 0; i < 4176; ++i){
__xlx_w24__tmp_vec.push_back(((Byte<2>*)__xlx_apatb_param_w24)[i]);
}
  int __xlx_size_param_w24 = 4176;
  int __xlx_offset_param_w24 = 0;
  int __xlx_offset_byte_param_w24 = 0*2;
  // Collect __xlx_w27__tmp_vec
std::vector<Byte<2>> __xlx_w27__tmp_vec;
for (size_t i = 0; i < 58; ++i){
__xlx_w27__tmp_vec.push_back(((Byte<2>*)__xlx_apatb_param_w27)[i]);
}
  int __xlx_size_param_w27 = 58;
  int __xlx_offset_param_w27 = 0;
  int __xlx_offset_byte_param_w27 = 0*2;
  // DUT call
  myproject(__xlx_apatb_param_cluster, __xlx_apatb_param_nModule, __xlx_apatb_param_x_local, __xlx_apatb_param_y_local, __xlx_apatb_param_layer29_out, __xlx_w9__tmp_vec.data(), __xlx_apatb_param_b9_0, __xlx_apatb_param_b9_1, __xlx_w16__tmp_vec.data(), __xlx_apatb_param_b16_0, __xlx_apatb_param_b16_1, __xlx_apatb_param_b16_2, __xlx_apatb_param_b16_3, __xlx_apatb_param_b16_4, __xlx_apatb_param_b16_5, __xlx_apatb_param_b16_6, __xlx_apatb_param_b16_7, __xlx_apatb_param_b16_8, __xlx_apatb_param_b16_9, __xlx_apatb_param_b16_10, __xlx_apatb_param_b16_11, __xlx_apatb_param_b16_12, __xlx_apatb_param_b16_13, __xlx_apatb_param_b16_14, __xlx_apatb_param_b16_15, __xlx_w21__tmp_vec.data(), __xlx_apatb_param_b21_0, __xlx_apatb_param_b21_1, __xlx_apatb_param_b21_2, __xlx_apatb_param_b21_3, __xlx_apatb_param_b21_4, __xlx_apatb_param_b21_5, __xlx_apatb_param_b21_6, __xlx_apatb_param_b21_7, __xlx_apatb_param_b21_8, __xlx_apatb_param_b21_9, __xlx_apatb_param_b21_10, __xlx_apatb_param_b21_11, __xlx_apatb_param_b21_12, __xlx_apatb_param_b21_13, __xlx_apatb_param_b21_14, __xlx_apatb_param_b21_15, __xlx_apatb_param_b21_16, __xlx_apatb_param_b21_17, __xlx_apatb_param_b21_18, __xlx_apatb_param_b21_19, __xlx_apatb_param_b21_20, __xlx_apatb_param_b21_21, __xlx_apatb_param_b21_22, __xlx_apatb_param_b21_23, __xlx_apatb_param_b21_24, __xlx_apatb_param_b21_25, __xlx_apatb_param_b21_26, __xlx_apatb_param_b21_27, __xlx_apatb_param_b21_28, __xlx_apatb_param_b21_29, __xlx_apatb_param_b21_30, __xlx_apatb_param_b21_31, __xlx_apatb_param_b21_32, __xlx_apatb_param_b21_33, __xlx_apatb_param_b21_34, __xlx_apatb_param_b21_35, __xlx_apatb_param_b21_36, __xlx_apatb_param_b21_37, __xlx_apatb_param_b21_38, __xlx_apatb_param_b21_39, __xlx_apatb_param_b21_40, __xlx_apatb_param_b21_41, __xlx_apatb_param_b21_42, __xlx_apatb_param_b21_43, __xlx_apatb_param_b21_44, __xlx_apatb_param_b21_45, __xlx_apatb_param_b21_46, __xlx_apatb_param_b21_47, __xlx_apatb_param_b21_48, __xlx_apatb_param_b21_49, __xlx_apatb_param_b21_50, __xlx_apatb_param_b21_51, __xlx_apatb_param_b21_52, __xlx_apatb_param_b21_53, __xlx_apatb_param_b21_54, __xlx_apatb_param_b21_55, __xlx_apatb_param_b21_56, __xlx_apatb_param_b21_57, __xlx_apatb_param_b21_58, __xlx_apatb_param_b21_59, __xlx_apatb_param_b21_60, __xlx_apatb_param_b21_61, __xlx_apatb_param_b21_62, __xlx_apatb_param_b21_63, __xlx_apatb_param_b21_64, __xlx_apatb_param_b21_65, __xlx_apatb_param_b21_66, __xlx_apatb_param_b21_67, __xlx_apatb_param_b21_68, __xlx_apatb_param_b21_69, __xlx_apatb_param_b21_70, __xlx_apatb_param_b21_71, __xlx_w24__tmp_vec.data(), __xlx_apatb_param_b24_0, __xlx_apatb_param_b24_1, __xlx_apatb_param_b24_2, __xlx_apatb_param_b24_3, __xlx_apatb_param_b24_4, __xlx_apatb_param_b24_5, __xlx_apatb_param_b24_6, __xlx_apatb_param_b24_7, __xlx_apatb_param_b24_8, __xlx_apatb_param_b24_9, __xlx_apatb_param_b24_10, __xlx_apatb_param_b24_11, __xlx_apatb_param_b24_12, __xlx_apatb_param_b24_13, __xlx_apatb_param_b24_14, __xlx_apatb_param_b24_15, __xlx_apatb_param_b24_16, __xlx_apatb_param_b24_17, __xlx_apatb_param_b24_18, __xlx_apatb_param_b24_19, __xlx_apatb_param_b24_20, __xlx_apatb_param_b24_21, __xlx_apatb_param_b24_22, __xlx_apatb_param_b24_23, __xlx_apatb_param_b24_24, __xlx_apatb_param_b24_25, __xlx_apatb_param_b24_26, __xlx_apatb_param_b24_27, __xlx_apatb_param_b24_28, __xlx_apatb_param_b24_29, __xlx_apatb_param_b24_30, __xlx_apatb_param_b24_31, __xlx_apatb_param_b24_32, __xlx_apatb_param_b24_33, __xlx_apatb_param_b24_34, __xlx_apatb_param_b24_35, __xlx_apatb_param_b24_36, __xlx_apatb_param_b24_37, __xlx_apatb_param_b24_38, __xlx_apatb_param_b24_39, __xlx_apatb_param_b24_40, __xlx_apatb_param_b24_41, __xlx_apatb_param_b24_42, __xlx_apatb_param_b24_43, __xlx_apatb_param_b24_44, __xlx_apatb_param_b24_45, __xlx_apatb_param_b24_46, __xlx_apatb_param_b24_47, __xlx_apatb_param_b24_48, __xlx_apatb_param_b24_49, __xlx_apatb_param_b24_50, __xlx_apatb_param_b24_51, __xlx_apatb_param_b24_52, __xlx_apatb_param_b24_53, __xlx_apatb_param_b24_54, __xlx_apatb_param_b24_55, __xlx_apatb_param_b24_56, __xlx_apatb_param_b24_57, __xlx_w27__tmp_vec.data(), __xlx_apatb_param_b27);
// print __xlx_apatb_param_w9
for (size_t i = 0; i < __xlx_size_param_w9; ++i) {
((Byte<2>*)__xlx_apatb_param_w9)[i] = __xlx_w9__tmp_vec[__xlx_offset_param_w9+i];
}
// print __xlx_apatb_param_w16
for (size_t i = 0; i < __xlx_size_param_w16; ++i) {
((Byte<2>*)__xlx_apatb_param_w16)[i] = __xlx_w16__tmp_vec[__xlx_offset_param_w16+i];
}
// print __xlx_apatb_param_w21
for (size_t i = 0; i < __xlx_size_param_w21; ++i) {
((Byte<2>*)__xlx_apatb_param_w21)[i] = __xlx_w21__tmp_vec[__xlx_offset_param_w21+i];
}
// print __xlx_apatb_param_w24
for (size_t i = 0; i < __xlx_size_param_w24; ++i) {
((Byte<2>*)__xlx_apatb_param_w24)[i] = __xlx_w24__tmp_vec[__xlx_offset_param_w24+i];
}
// print __xlx_apatb_param_w27
for (size_t i = 0; i < __xlx_size_param_w27; ++i) {
((Byte<2>*)__xlx_apatb_param_w27)[i] = __xlx_w27__tmp_vec[__xlx_offset_param_w27+i];
}
}
