#include "dag_reservation.cpp"

static std::vector<int64_t> chronological_rank(const DagProblem&b,const std::vector<int64_t>&a,const std::vector<int64_t>&r,int decoder){
 int n=a.size();std::vector<double>starts(n);std::vector<int>order(n);std::vector<int64_t>rank(n);
 if(decoder)dag_serial(b,a.data(),r.data(),starts.data(),nullptr);
 else dag_events(b,a.data(),r.data(),nullptr,starts.data(),nullptr);
 for(int i=0;i<n;++i)order[i]=i;
 std::sort(order.begin(),order.end(),[&](int x,int y){return starts[x]==starts[y]?r[x]<r[y]:starts[x]<starts[y];});
 for(int pos=0;pos<n;++pos)rank[order[pos]]=pos;return rank;
}

extern "C" double dag_bridge_search(const double*q,const uint64_t*d,int64_t*a,int64_t*r,int j,int o,int h,int g,int k,
 int steps,int mode,double temperature,uint64_t seed,double seconds,int*chosen_mode,int64_t*count){
 auto begin=std::chrono::steady_clock::now();
 auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
 DagProblem b(q,d,j,o,h,g,k);int n=j*o;
 auto score=[&](const std::vector<int64_t>&aa,const std::vector<int64_t>&rr,int decoder){return decoder?dag_serial(b,aa.data(),rr.data(),nullptr,nullptr):dag_events(b,aa.data(),rr.data(),nullptr,nullptr,nullptr);};
 std::vector<int64_t>cur_a(a,a+n),cur_r(r,r+n);
 int decoder=mode==2?0:mode;double cost=score(cur_a,cur_r,decoder);
 if(mode!=0){
  if(mode==2){double value=score(cur_a,cur_r,1);if(value<cost){cost=value;decoder=1;}}
  auto mapped=chronological_rank(b,cur_a,cur_r,0);double value=score(cur_a,mapped,1);
  if(value<cost){cost=value;cur_r=mapped;decoder=1;}
 }
 auto best_a=cur_a,best_r=cur_r;int best_mode=decoder;double best=cost,start=cost,reserve=.001;
 int64_t scored=0;std::mt19937_64 rng(seed);
 for(int step=0;step<steps;++step){
  double tick=elapsed();if(seconds>0&&tick+reserve>=seconds)break;
  int candidate_mode=mode==2?int(rng()%2):mode;
  auto aa=cur_a,rr=candidate_mode==decoder?cur_r:chronological_rank(b,cur_a,cur_r,decoder);
  auto flip=[&](int i){for(int host=0;host<h;++host)if(host!=aa[i]&&std::isfinite(b.m.b.p[i*h+host])){aa[i]=host;break;}};
  int x=rng()%n,y=rng()%n,action=rng()%4;
  if(action==0)flip(x);
  else if(action==1)std::swap(rr[x],rr[y]);
  else if(action==2){flip(x);std::swap(rr[x],rr[y]);}
  else{int length=2+rng()%7;for(int z=0;z<length;++z){int v=rng()%n;std::swap(rr[x],rr[v]);if(rng()%2)flip(v);}}
  double value=score(aa,rr,candidate_mode);
  double progress=seconds>0?std::min(1.,tick/seconds):double(step)/std::max(1,steps-1);
  double temp=temperature*start*std::pow(.01,progress),u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(value<cost||(temperature>0&&u<std::exp((cost-value)/temp))){cur_a=aa;cur_r=rr;cost=value;decoder=candidate_mode;}
  if(value<best){best=value;best_a=aa;best_r=rr;best_mode=candidate_mode;}
  ++scored;reserve=std::max(reserve,2*(elapsed()-tick));
 }
 std::copy(best_a.begin(),best_a.end(),a);std::copy(best_r.begin(),best_r.end(),r);*chosen_mode=best_mode;*count=scored;return best;
}
