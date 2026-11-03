#include "normal_mode_switch.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace robojudo::normal_mode;
struct Fake {
  double time = 0;
  int initial = 801;
  int sends = 0, reads = 0, post_reads = 0;
  int request_code = 0;
  bool throw_start = false, stale = false, moving = false;
  bool finite = true, v2 = false;
  int fail_read = -1, throw_read = -1;
  double query_delay = 0;
  std::vector<int> post{500,500,500};
  std::function<int(Fake&)> pre;
  std::function<MotionSample(Fake&)> motion;
  Result run() {
    Limits l;
    return Run([&](int32_t& id) {
      ++reads; time += query_delay;
      if (reads == throw_read) throw std::runtime_error("RPC exception");
      if (reads == fail_read) return 42;
      if (sends) id=post.at(std::min<std::size_t>(post_reads++,post.size()-1));
      else id=pre ? pre(*this) : initial;
      return 0;
    }, [&]() {
      ++sends;
      if (throw_start) throw std::runtime_error("request timeout");
      return request_code;
    }, [&]() {
      if (motion) return motion(*this);
      MotionSample s; s.fresh=!stale;
      s.xy_speed=moving ? .2 : 0;
      if (!finite) s.angular_speed=std::numeric_limits<double>::quiet_NaN();
      return s;
    }, [&](int32_t id) { return id==801 || (v2 && id==802); },
    [&]() {return time;}, [&](double s) {time+=s;}, l);
  }
};
int main() {
  int cases=0;
  { Fake f; auto r=f.run();assert(r.code==OK && r.target_confirmed && f.sends==1 && f.post_reads==3 && f.time>=.34);++cases; }
  { Fake f;f.initial=500;auto r=f.run();assert(r.code==OK && !r.request_sent && f.sends==0 && f.reads==4);++cases; }
  for(int source: {0,1,4,501,802,1000}) {Fake f;f.initial=source;auto r=f.run();assert(r.code==WRONG_SOURCE && f.sends==0);++cases;}
  { Fake f;f.initial=802;f.v2=true;assert(f.run().code==OK && f.sends==1);++cases; }
  { Fake f;f.fail_read=1;auto r=f.run();assert(r.code==QUERY_FAILED && r.observed_fsm==-1 && f.sends==0);++cases; }
  { Fake f;f.throw_read=1;assert(f.run().code==QUERY_FAILED && f.sends==0);++cases; }
  { Fake f;f.stale=true;assert(f.run().code==STALE_STATE && f.sends==0);++cases; }
  { Fake f;f.moving=true;assert(f.run().code==NOT_STATIONARY && f.sends==0);++cases; }
  { Fake f;f.finite=false;assert(f.run().code==NOT_STATIONARY && f.sends==0);++cases; }
  { Fake f;f.pre=[](Fake& x){return x.time>.15 ? 1000 : 801;};assert(f.run().code==INTERRUPTED && f.sends==0);++cases; }
  { Fake f;f.request_code=88;f.post={801};auto r=f.run();assert(r.code==REQUEST_FAILED && r.sdk_request_result==88 && f.sends==1 && !r.target_confirmed);++cases; }
  { Fake f;f.request_code=88;auto r=f.run();assert(r.code==OK && r.sdk_request_result==88 && r.request_returned && f.sends==1);++cases; }
  { Fake f;f.throw_start=true;auto r=f.run();assert(r.code==OK && r.request_sent && !r.request_returned && f.sends==1);++cases; }
  { Fake f;f.throw_start=true;f.post={801};assert(f.run().code==REQUEST_FAILED && f.sends==1);++cases; }
  { Fake f;f.post={801};assert(f.run().code==CONFIRM_TIMEOUT && f.sends==1);++cases; }
  for(int target:{1,501,1000}) {Fake f;f.post={target};auto r=f.run();assert(r.code==INTERRUPTED && r.observed_fsm==target && f.sends==1);++cases;}
  { Fake f;f.post={500,801,500};assert(f.run().code==INTERRUPTED && f.sends==1);++cases; }
  { Fake f;f.post={500,500,1};auto r=f.run();assert(r.code==INTERRUPTED && !r.target_confirmed);++cases; }
  { Fake f;f.fail_read=18;auto r=f.run();assert(r.code==QUERY_FAILED && r.observed_fsm==-1 && f.sends==1);++cases; }
  { Fake f;f.pre=[](Fake& x){return x.reads==3 ? 801 : 500;};assert(f.run().code==INTERRUPTED && f.sends==0);++cases; }
  { Fake f;f.motion=[](Fake& x){MotionSample s;s.fresh=true;s.xy_speed=(x.time>.18 && x.time<.3) ? .2 : 0;return s;};auto r=f.run();assert(r.code==OK && r.elapsed_s>=.64 && f.sends==1);++cases; }
  { Fake f;f.query_delay=2.5;assert(f.run().code==NOT_STATIONARY && f.sends==0);++cases; }
  { Fake f;f.initial=500;f.query_delay=1.1;assert(f.run().code==CONFIRM_TIMEOUT && f.sends==0);++cases; }
  std::cout<<cases<<" offline transition scenarios passed\n";
}
