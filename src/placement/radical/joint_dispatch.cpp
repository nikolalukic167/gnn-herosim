#include "dispatch_adaptive.cpp"
#include <chrono>

static double joint_optimize(const double*q,int64_t*a,int64_t*r,
 int j,int o,int h,int g,int k,int steps,int mode,int anneal,uint64_t seed,double seconds,int64_t*count){
 auto begin=std::chrono::steady_clock::now();
 auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
 Mixed m(Problem(q,j,o,h,g,k,7));int n=j*o;
 std::vector<int64_t>cur_a(a,a+n),cur_r(r,r+n),best_a(cur_a),best_r(cur_r);
 std::mt19937_64 rng(seed);
 double cost=priority_replay(m,cur_a.data(),cur_r.data()),best=cost,start=cost;
 double reserve=.001;int scored=0;
 for(int step=0;step<steps;++step){
  double tick=elapsed();if(seconds>0&&tick+reserve>=seconds)break;
  auto next_a=cur_a,next_r=cur_r;
  auto flip=[&](int i){for(int host=0;host<h;++host)if(host!=next_a[i]&&std::isfinite(m.b.p[i*h+host])){next_a[i]=host;break;}};
  int x=rng()%n,y=rng()%n;if(y==x)y=(y+1)%n;
  int action=mode==0?0:mode==1?1:mode==2?int(rng()%3):int(rng()%4);
  if(action==0)flip(x);
  else if(action==1)std::swap(next_r[x],next_r[y]);
  else if(action==2){flip(x);if(rng()%2)flip(y);std::swap(next_r[x],next_r[y]);}
  else{
   int job=x/o,first=x%o,length=std::min(o-first,2<<int(rng()%3));
   std::vector<int>block;for(int z=first;z<first+length;++z){int i=job*o+z;block.push_back(i);if(rng()%2)flip(i);}
   std::sort(block.begin(),block.end(),[&](int l,int rr){return cur_r[l]<cur_r[rr];});
   std::vector<int>order;for(int i=0;i<n;++i)if(std::find(block.begin(),block.end(),i)==block.end())order.push_back(i);
   std::sort(order.begin(),order.end(),[&](int l,int rr){return cur_r[l]<cur_r[rr];});
   int at=rng()%(order.size()+1);order.insert(order.begin()+at,block.begin(),block.end());
   for(int pos=0;pos<n;++pos)next_r[order[pos]]=pos;
  }
  double value=priority_replay(m,next_a.data(),next_r.data());
  double progress=seconds>0?std::min(1.,tick/seconds):double(step)/std::max(1,steps-1);
  double temp=.025*start*std::pow(.01,progress);
  double u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(value<cost||(anneal&&u<std::exp((cost-value)/temp))){cur_a=next_a;cur_r=next_r;cost=value;}
  if(value<best){best=value;best_a=next_a;best_r=next_r;}
  ++scored;reserve=std::max(reserve,2*(elapsed()-tick));
 }
 if(count)*count=scored;
 std::copy(best_a.begin(),best_a.end(),a);std::copy(best_r.begin(),best_r.end(),r);return best;
}

extern "C" double joint_dispatch_search(const double*q,int64_t*a,int64_t*r,
 int j,int o,int h,int g,int k,int steps,int mode,int anneal,uint64_t seed){
 return joint_optimize(q,a,r,j,o,h,g,k,steps,mode,anneal,seed,0,nullptr);
}

extern "C" double joint_dispatch_timed(const double*q,int64_t*a,int64_t*r,
 int j,int o,int h,int g,int k,int mode,int anneal,uint64_t seed,double seconds,int64_t*count){
 return joint_optimize(q,a,r,j,o,h,g,k,1000000000,mode,anneal,seed,seconds,count);
}
