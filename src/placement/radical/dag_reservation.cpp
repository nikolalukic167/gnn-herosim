#include "mixed.cpp"
#include <chrono>

struct DagProblem {
 Mixed m;
 const uint64_t* pred;
 explicit DagProblem(const double*q,const uint64_t*d,int j,int o,int h,int g,int k):m(Problem(q,j,o,h,g,k,7)),pred(d){}
};

static double dag_serial(const DagProblem&d,const int64_t*a,const int64_t*r,double*starts,double*ends){
 const auto&b=d.m.b;int n=b.J*b.O;
 std::vector<uint64_t>scheduled(b.J,0);
 std::vector<double>host(b.H,0),lock(28,0),slots(2*b.G,0),completion(n,0);
 std::vector<int>last(b.H,-1);
 for(int step=0;step<n;++step){
  int chosen=-1;
  for(int i=0;i<n;++i){int j=i/b.O,k=i%b.O;
   if(!(scheduled[j]&(uint64_t(1)<<k))&&(scheduled[j]&d.pred[i])==d.pred[i]&&(chosen<0||r[i]<r[chosen]))chosen=i;
  }
  if(chosen<0)return std::numeric_limits<double>::infinity();
  int i=chosen,j=i/b.O,k=i%b.O,h=a[i];double t=host[h];
  uint64_t parents=d.pred[i];while(parents){int z=__builtin_ctzll(parents);t=std::max(t,completion[j*b.O+z]);parents&=parents-1;}
  int bits=d.m.locks[i];while(bits){int z=__builtin_ctz(bits);t=std::max(t,lock[z]);bits&=bits-1;}
  for(int g=0;g<b.G;++g)if(b.domains[h*b.G+g])t=std::max(t,std::min(slots[2*g],slots[2*g+1]));
  double end=t+b.p[i*b.H+h]+(last[h]>=0?b.setup[last[h]*b.K+d.m.types[i]]:0.);
  host[h]=end;last[h]=d.m.types[i];completion[i]=end;scheduled[j]|=uint64_t(1)<<k;
  bits=d.m.locks[i];while(bits){int z=__builtin_ctz(bits);lock[z]=end;bits&=bits-1;}
  for(int g=0;g<b.G;++g)if(b.domains[h*b.G+g]){int slot=2*g+(slots[2*g+1]<slots[2*g]);slots[slot]=end;}
  if(starts)starts[i]=t;if(ends)ends[i]=end;
 }
 double total=0;for(int j=0;j<b.J;++j)total+=completion[(j+1)*b.O-1];return total;
}

static double dag_events(const DagProblem&d,const int64_t*a,const int64_t*r,const double*release,double*starts,double*ends){
 const auto&b=d.m.b;int n=b.J*b.O,done=0,held=0;uint32_t hosts=0;
 std::vector<int>state(n,0),running(b.H,-1),last(b.H,-1);
 std::vector<uint64_t>completed(b.J,0);
 std::vector<double>finish(b.H,0);double t=0,total=0;
 while(done<n){
  for(int h=0;h<b.H;++h)if(running[h]>=0&&finish[h]<=t+1e-9){
   int i=running[h],j=i/b.O,k=i%b.O;state[i]=2;completed[j]|=uint64_t(1)<<k;
   last[h]=d.m.types[i];running[h]=-1;held&=~d.m.locks[i];hosts&=~(uint32_t(1)<<h);++done;
   if(ends)ends[i]=t;if(k==b.O-1)total+=t;
  }
  if(done==n)break;
  std::vector<int>ready;
  for(int i=0;i<n;++i)if(state[i]==0&&(completed[i/b.O]&d.pred[i])==d.pred[i])ready.push_back(i);
  std::sort(ready.begin(),ready.end(),[&](int x,int y){return r[x]<r[y];});
  for(int i:ready){int h=a[i];if((release&&release[i]>t+1e-9)||running[h]>=0||(held&d.m.locks[i]))continue;
   bool blocked=false;for(int g=0;g<b.G;++g)if((d.m.domains[g]&(uint32_t(1)<<h))&&__builtin_popcount(d.m.domains[g]&hosts)>=2){blocked=true;break;}
   if(blocked)continue;
   state[i]=1;running[h]=i;held|=d.m.locks[i];hosts|=uint32_t(1)<<h;
   finish[h]=t+b.p[i*b.H+h]+(last[h]>=0?b.setup[last[h]*b.K+d.m.types[i]]:0.);
   if(starts)starts[i]=t;
  }
  double next=std::numeric_limits<double>::infinity();
  for(int h=0;h<b.H;++h)if(running[h]>=0)next=std::min(next,finish[h]);
  if(release)for(int i:ready)if(state[i]==0&&release[i]>t+1e-9)next=std::min(next,release[i]);
  if(!std::isfinite(next))return next;t=next;
 }
 return total;
}

extern "C" double dag_plan(const double*q,const uint64_t*d,int64_t*a,int64_t*r,int j,int o,int h,int g,int k,int mode,double*starts,double*ends){
 DagProblem b(q,d,j,o,h,g,k);return mode?dag_serial(b,a,r,starts,ends):dag_events(b,a,r,nullptr,starts,ends);
}
extern "C" double dag_replay(const double*q,const uint64_t*d,int64_t*a,int64_t*r,const double*release,int j,int o,int h,int g,int k,double*starts,double*ends){
 return dag_events(DagProblem(q,d,j,o,h,g,k),a,r,release,starts,ends);
}

extern "C" double dag_search(const double*q,const uint64_t*d,int64_t*a,int64_t*r,int j,int o,int h,int g,int k,
 int steps,int mode,double temperature,uint64_t seed,double seconds,int*chosen_mode,int64_t*count){
 auto begin=std::chrono::steady_clock::now();
 auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
 DagProblem b(q,d,j,o,h,g,k);int n=j*o;
 auto score=[&](const std::vector<int64_t>&aa,const std::vector<int64_t>&rr,int decoder){return decoder?dag_serial(b,aa.data(),rr.data(),nullptr,nullptr):dag_events(b,aa.data(),rr.data(),nullptr,nullptr,nullptr);};
 std::vector<int64_t>cur_a(a,a+n),cur_r(r,r+n),best_a(cur_a),best_r(cur_r),alt(n);
 for(int i=0;i<n;++i)for(int host=0;host<h;++host)if(host!=a[i]&&std::isfinite(b.m.b.p[i*h+host]))alt[i]=host;
 int decoder=mode==2?0:mode;double cost=score(cur_a,cur_r,decoder);
 if(mode==2){double serial=score(cur_a,cur_r,1);if(serial<cost){decoder=1;cost=serial;}}
 double best=cost,start=cost,reserve=.001;int best_mode=decoder;int64_t scored=0;std::mt19937_64 rng(seed);
 for(int step=0;step<steps;++step){
  double tick=elapsed();if(seconds>0&&tick+reserve>=seconds)break;
  auto aa=cur_a,rr=cur_r;
  auto flip=[&](int i){for(int host=0;host<h;++host)if(host!=aa[i]&&std::isfinite(b.m.b.p[i*h+host])){aa[i]=host;break;}};
  int x=rng()%n,y=rng()%n,action=rng()%4;
  if(action==0)flip(x);
  else if(action==1)std::swap(rr[x],rr[y]);
  else if(action==2){flip(x);std::swap(rr[x],rr[y]);}
  else{int length=2+rng()%7;for(int z=0;z<length;++z){int v=rng()%n;std::swap(rr[x],rr[v]);if(rng()%2)flip(v);}}
  int candidate_mode=mode==2?int(rng()%2):mode;double value=score(aa,rr,candidate_mode);
  double progress=seconds>0?std::min(1.,tick/seconds):double(step)/std::max(1,steps-1);
  double temp=temperature*start*std::pow(.01,progress),u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(value<cost||(temperature>0&&u<std::exp((cost-value)/temp))){cur_a=aa;cur_r=rr;cost=value;decoder=candidate_mode;}
  if(value<best){best=value;best_a=aa;best_r=rr;best_mode=candidate_mode;}
  ++scored;reserve=std::max(reserve,2*(elapsed()-tick));
 }
 std::copy(best_a.begin(),best_a.end(),a);std::copy(best_r.begin(),best_r.end(),r);
 *chosen_mode=best_mode;*count=scored;return best;
}
