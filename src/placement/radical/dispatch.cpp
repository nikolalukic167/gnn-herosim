#include "mixed.cpp"

static double priority_replay(const Mixed&m,const int64_t*a,const int64_t*rank,double*ends=nullptr){
 const auto&b=m.b;int next[64]={},running[32],last[32];double ready[64]={},finish[32]={};
 std::fill(running,running+b.H,-1);std::fill(last,last+b.H,-1);
 uint64_t busy=0;uint32_t hosts=0;int held=0,done=0;double t=0,total=0;
 while(done<b.J*b.O){
  for(int h=0;h<b.H;++h)if(running[h]>=0&&finish[h]<=t+1e-9){
   int i=running[h],j=i/b.O;running[h]=-1;last[h]=m.types[i];++next[j];++done;ready[j]=t;
   busy&=~(uint64_t(1)<<j);hosts&=~(uint32_t(1)<<h);held&=~m.locks[i];
   if(ends)ends[i]=t;if(next[j]==b.O)total+=t;
  }
  if(done==b.J*b.O)break;
  std::vector<int> candidates;
  for(int j=0;j<b.J;++j)if(!(busy&(uint64_t(1)<<j))&&next[j]<b.O)candidates.push_back(j);
  std::sort(candidates.begin(),candidates.end(),[&](int x,int y){auto rx=rank[x*b.O+next[x]],ry=rank[y*b.O+next[y]];return rx==ry?x<y:rx<ry;});
  for(int j:candidates){int i=j*b.O+next[j],h=a[i];if(running[h]>=0||(held&m.locks[i]))continue;
   bool blocked=false;for(int g=0;g<b.G;++g)if((m.domains[g]&(uint32_t(1)<<h))&&__builtin_popcount(m.domains[g]&hosts)>=2){blocked=true;break;}
   if(blocked)continue;
   running[h]=i;finish[h]=t+b.p[i*b.H+h]+(last[h]>=0?b.setup[last[h]*b.K+m.types[i]]:0.);
   busy|=uint64_t(1)<<j;hosts|=uint32_t(1)<<h;held|=m.locks[i];
  }
  double nt=std::numeric_limits<double>::infinity();for(int h=0;h<b.H;++h)if(running[h]>=0)nt=std::min(nt,finish[h]);
  if(!std::isfinite(nt))return nt;t=nt;
 }
 return total;
}
extern "C" double dispatch_score(const double*q,const int64_t*a,const int64_t*r,int j,int o,int h,int g,int k,double*ends){return priority_replay(Mixed(Problem(q,j,o,h,g,k,7)),a,r,ends);}
extern "C" double dispatch_search(const double*q,const int64_t*a,int64_t*r,int j,int o,int h,int g,int k,int steps,uint64_t seed){
 Mixed b(Problem(q,j,o,h,g,k,7));int n=j*o;std::vector<int64_t>cur(r,r+n),best(cur);std::mt19937_64 rng(seed);
 double c=priority_replay(b,a,cur.data()),bc=c,start=c;
 for(int step=0;step<steps;++step){int x=rng()%n,y=rng()%n;std::swap(cur[x],cur[y]);double v=priority_replay(b,a,cur.data());
  double temp=.025*start*std::pow(.01,double(step)/std::max(1,steps-1));double u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(v<c||u<std::exp((c-v)/temp)){c=v;if(v<bc){bc=v;best=cur;}}else std::swap(cur[x],cur[y]);
 }
 std::copy(best.begin(),best.end(),r);return bc;
}
