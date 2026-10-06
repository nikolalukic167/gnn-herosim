#include "joint_dispatch.cpp"

extern "C" double region_dispatch_repair(const double*q,int64_t*a,int64_t*r,
 int j,int o,int h,int g,int t,const int64_t*region,int k){
 Mixed m(Problem(q,j,o,h,g,t,7));int n=j*o;
 std::vector<int64_t>base_a(a,a+n),base_r(r,r+n),best_a(base_a),best_r(base_r);
 std::vector<int64_t>tokens;for(int i=0;i<k;++i)tokens.push_back(r[region[i]]);
 std::sort(tokens.begin(),tokens.end());
 double best=priority_replay(m,a,r);
 for(int mask=0;mask<(1<<k);++mask){
  auto next_a=base_a;
  for(int z=0;z<k;++z)if(mask&(1<<z)){
   int i=region[z];
   for(int host=0;host<h;++host)if(host!=base_a[i]&&std::isfinite(m.b.p[i*h+host])){next_a[i]=host;break;}
  }
  auto permutation=tokens;
  do{
   auto next_r=base_r;
   for(int z=0;z<k;++z)next_r[region[z]]=permutation[z];
   double cost=priority_replay(m,next_a.data(),next_r.data());
   if(cost<best){best=cost;best_a=next_a;best_r=next_r;}
  }while(std::next_permutation(permutation.begin(),permutation.end()));
 }
 std::copy(best_a.begin(),best_a.end(),a);std::copy(best_r.begin(),best_r.end(),r);
 return best;
}
