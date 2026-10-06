#include "kernel.cpp"

struct Mixed {
 Problem b;
 int locks[4096],types[4096];
 uint32_t domains[32];
 explicit Mixed(const Problem& p):b(p){
  for(int i=0;i<b.J*b.O;++i){locks[i]=(int)b.locks[i];types[i]=(int)b.types[i];}
  for(int g=0;g<b.G;++g){domains[g]=0;for(int h=0;h<b.H;++h)if(b.domains[h*b.G+g])domains[g]|=uint32_t(1)<<h;}
 }
};
static double mixed_replay(const Mixed&m,const int64_t*a,double* ends=nullptr){
 const auto&b=m.b;int next[64]={},running[32],last[32];double ready[64]={},finish[32]={};
 std::fill(running,running+b.H,-1);std::fill(last,last+b.H,-1);
 uint64_t busyjobs=0;uint32_t hosts=0;int held=0,done=0;double t=0,total=0;
 while(done<b.J*b.O){
  for(int h=0;h<b.H;++h)if(running[h]>=0 && finish[h]<=t+1e-9){
   int i=running[h],j=i/b.O;running[h]=-1;last[h]=m.types[i];ready[j]=t;++next[j];++done;
   busyjobs&=~(uint64_t(1)<<j);hosts&=~(uint32_t(1)<<h);held&=~m.locks[i];
   if(ends)ends[i]=t;if(next[j]==b.O)total+=t;
  }
  if(done==b.J*b.O)break;
  int heads[32],count=0;
  for(int h=0;h<b.H;++h)if(running[h]<0){
   int best=-1;
   for(int j=0;j<b.J;++j)if(!(busyjobs&(uint64_t(1)<<j)) && next[j]<b.O && a[j*b.O+next[j]]==h)
    if(best<0 || ready[j]<ready[best] || (ready[j]==ready[best]&&j<best))best=j;
   if(best>=0)heads[count++]=best;
  }
  std::sort(heads,heads+count,[&](int x,int y){return ready[x]==ready[y]?x<y:ready[x]<ready[y];});
  for(int ii=0;ii<count;++ii){int j=heads[ii],i=j*b.O+next[j],h=a[i];
   if(held&m.locks[i])continue;
   bool blocked=false;
   for(int g=0;g<b.G;++g)if((m.domains[g]&(uint32_t(1)<<h)) && __builtin_popcount(m.domains[g]&hosts)>=2){blocked=true;break;}
   if(blocked)continue;
   double dur=b.p[i*b.H+h]+(last[h]>=0?b.setup[last[h]*b.K+m.types[i]]:0.);
   running[h]=i;finish[h]=t+dur;held|=m.locks[i];hosts|=uint32_t(1)<<h;busyjobs|=uint64_t(1)<<j;
  }
  double nt=std::numeric_limits<double>::infinity();for(int h=0;h<b.H;++h)if(running[h]>=0)nt=std::min(nt,finish[h]);
  if(!std::isfinite(nt))return nt;t=nt;
 }
 return total;
}
extern "C" double mixed_score(const double*q,const int64_t*a,int j,int o,int h,int g,int k,double*ends){return mixed_replay(Mixed(Problem(q,j,o,h,g,k,7)),a,ends);}
extern "C" double mixed_search(const double*q,int64_t*a,int j,int o,int h,int g,int k,int steps,int anneal,uint64_t seed){
 Mixed b(Problem(q,j,o,h,g,k,7));int n=j*o;std::vector<int64_t>cur(a,a+n),best(cur),alt(n);std::mt19937_64 rng(seed);
 double score=mixed_replay(b,cur.data()),bestscore=score,start=score;
 for(int i=0;i<n;++i)for(int host=0;host<h;++host)if(std::isfinite(b.b.p[i*h+host])&&host!=cur[i])alt[i]=host;
 if(!anneal){for(int r=0;r<steps;++r){double candidate=score;int chosen=-1;
  for(int i=0;i<n;++i){std::swap(cur[i],alt[i]);double s=mixed_replay(b,cur.data());std::swap(cur[i],alt[i]);if(s<candidate){candidate=s;chosen=i;}}
  if(chosen<0)break;std::swap(cur[chosen],alt[chosen]);score=candidate;best=cur;bestscore=score;
 }}else{for(int r=0;r<steps;++r){int i=rng()%n;std::swap(cur[i],alt[i]);double s=mixed_replay(b,cur.data());double temp=.025*start*std::pow(.01,double(r)/std::max(1,steps-1));double u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(s<score || u<std::exp((score-s)/temp)){score=s;if(s<bestscore){bestscore=s;best=cur;}}else std::swap(cur[i],alt[i]);
 }}std::copy(best.begin(),best.end(),a);return bestscore;
}
