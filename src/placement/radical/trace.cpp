#include "mixed.cpp"
#include <fstream>
#include <iomanip>
extern "C" double mixed_trace(const double*q,int64_t*a,int j,int o,int h,int g,int k,int steps,uint64_t seed,const char*path){
 std::ofstream out(path,std::ios::app);if(!out)return std::numeric_limits<double>::quiet_NaN();out<<std::setprecision(17);
 auto record=[&](const std::vector<int64_t>&plan,double cost){out<<"{\"placement_plan\":[";for(int x=0;x<j;++x){if(x)out<<',';out<<'[';for(int y=0;y<o;++y){if(y)out<<',';out<<plan[x*o+y];}out<<']';}out<<"],\"rtt_ms\":"<<cost<<"}\n";};
 Mixed b(Problem(q,j,o,h,g,k,7));int n=j*o;std::vector<int64_t>cur(a,a+n),best(cur),alt(n);std::mt19937_64 rng(seed);double score=mixed_replay(b,cur.data()),bestscore=score,start=score;record(cur,score);
 for(int i=0;i<n;++i)for(int host=0;host<h;++host)if(std::isfinite(b.b.p[i*h+host])&&host!=cur[i])alt[i]=host;
 for(int r=0;r<steps;++r){int i=rng()%n;std::swap(cur[i],alt[i]);double s=mixed_replay(b,cur.data());record(cur,s);double temp=.025*start*std::pow(.01,double(r)/std::max(1,steps-1));double u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(s<score||u<std::exp((score-s)/temp)){score=s;if(s<bestscore){bestscore=s;best=cur;}}else std::swap(cur[i],alt[i]);
 }
 std::copy(best.begin(),best.end(),a);out.flush();return out?bestscore:std::numeric_limits<double>::quiet_NaN();
}
