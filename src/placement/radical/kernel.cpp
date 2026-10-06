#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
// q: processing[N,H], types[N], locks[N], domains[H,G], thermal[H,H],
// maintenance[H,2], deadlines[J], power[N], setup[K,K].
struct Problem {
 const double *p,*types,*locks,*domains,*thermal,*maintenance,*deadlines,*power,*setup;
 int J,O,H,G,K,flags;
 Problem(const double*q,int j,int o,int h,int g,int k,int f):J(j),O(o),H(h),G(g),K(k),flags(f){
 int n=J*O;p=q;types=p+n*H;locks=types+n;domains=locks+n;thermal=domains+H*G;
 maintenance=thermal+H*H;deadlines=maintenance+H*2;power=deadlines+J;setup=power+n;
 }
};
static double simulate(const Problem&b,const int64_t*a,double* ends=nullptr){
 int n=b.J*b.O;std::vector<int> next(b.J,0),running(b.H,-1),last(b.H,-1);
 std::vector<double> ready(b.J,0),finish(b.H,0),heat(b.H,0),completion(b.J,0);
 double t=0,energy=0;int done=0,guard=0;
 while(done<n){
  if(++guard>100000)return std::numeric_limits<double>::infinity();
  for(int h=0;h<b.H;++h)if(running[h]>=0 && finish[h]<=t+1e-9){
   int i=running[h],j=i/b.O;running[h]=-1;last[h]=(int)b.types[i];ready[j]=t;++next[j];++done;
   if(ends)ends[i]=t;if(next[j]==b.O)completion[j]=t;
  }
  if(done==n)break;
  bool changed=true;
  while(changed){changed=false;
   // Global FIFO among machine heads; deterministic job-id tie break.
   std::vector<int> candidates;
   for(int h=0;h<b.H;++h)if(running[h]<0){
    int best=-1;
    for(int j=0;j<b.J;++j)if(next[j]<b.O){int i=j*b.O+next[j];bool active=false;
     for(int x:running)if(x==i)active=true;
     if(!active && a[i]==h && (best<0 || ready[j]<ready[best] || (ready[j]==ready[best]&&j<best)))best=j;
    }
    if(best>=0)candidates.push_back(best);
   }
   std::sort(candidates.begin(),candidates.end(),[&](int x,int y){return ready[x]==ready[y]?x<y:ready[x]<ready[y];});
   for(int j:candidates){int i=j*b.O+next[j],h=a[i];bool blocked=false;
    if(b.flags&2)for(int x:running)if(x>=0 && ((int)b.locks[x] & (int)b.locks[i]))blocked=true;
    if(b.flags&4)for(int g=0;g<b.G;++g)if(b.domains[h*b.G+g]){int count=0;for(int hh=0;hh<b.H;++hh)if(running[hh]>=0 && b.domains[hh*b.G+g])++count;if(count>=2)blocked=true;}
    if(blocked)continue;
    double dur=b.p[i*b.H+h];
    if((b.flags&1)&&last[h]>=0)dur+=b.setup[last[h]*b.K+(int)b.types[i]];
    if(b.flags&8)dur*=1.+.04*heat[h];
    double lo=b.maintenance[h*2],hi=b.maintenance[h*2+1];
    if((b.flags&16)&& t<hi-1e-9 && t+dur>lo+1e-9)continue;
    running[h]=i;finish[h]=t+dur;changed=true;
    // Piecewise price known at dispatch; energy is charged for the whole non-preemptive operation.
    double price=1.+((int)(t/40.)+h)%3;
    energy+=dur*b.power[i]*price;
   }
  }
  double nt=std::numeric_limits<double>::infinity();
  for(int h=0;h<b.H;++h){if(running[h]>=0)nt=std::min(nt,finish[h]);if((b.flags&16)&&b.maintenance[h*2+1]>t+1e-9)nt=std::min(nt,b.maintenance[h*2+1]);}
  if(!std::isfinite(nt))return nt;
  double decay=std::exp(-(nt-t)/25.);
  for(int h=0;h<b.H;++h){double input=0;for(int hh=0;hh<b.H;++hh)if(running[hh]>=0)input+=b.thermal[h*b.H+hh]*b.power[running[hh]];heat[h]=heat[h]*decay+input*25*(1-decay);}
  t=nt;
 }
 double total=0;for(int j=0;j<b.J;++j){total+=completion[j];if(b.flags&32)total+=3*std::max(0.,completion[j]-b.deadlines[j]);}
 if(b.flags&64)total+=.2*energy;
 return total;
}
extern "C" double radical_score(const double*q,const int64_t*a,int j,int o,int h,int g,int k,int f,double*ends){return simulate(Problem(q,j,o,h,g,k,f),a,ends);}
extern "C" double radical_search(const double*q,int64_t*a,int j,int o,int h,int g,int k,int f,int steps,int anneal,uint64_t seed){
 Problem b(q,j,o,h,g,k,f);int n=j*o;std::vector<int64_t>cur(a,a+n),best(cur),alt(n);std::mt19937_64 rng(seed);
 double score=simulate(b,cur.data()),bestscore=score,start=score;
 for(int i=0;i<n;++i)for(int host=0;host<h;++host)if(std::isfinite(b.p[i*h+host])&&host!=cur[i])alt[i]=host;
 if(!anneal){for(int r=0;r<steps;++r){double candidate=score;int chosen=-1;
  for(int i=0;i<n;++i){std::swap(cur[i],alt[i]);double s=simulate(b,cur.data());std::swap(cur[i],alt[i]);if(s<candidate){candidate=s;chosen=i;}}
  if(chosen<0)break;std::swap(cur[chosen],alt[chosen]);score=candidate;best=cur;bestscore=score;
 }}else{for(int r=0;r<steps;++r){int i=rng()%n;std::swap(cur[i],alt[i]);double s=simulate(b,cur.data());double temp=.025*start*std::pow(.01,double(r)/std::max(1,steps-1));double u=(rng()+.5)/(double(std::numeric_limits<uint64_t>::max())+1.);
  if(s<score || u<std::exp((score-s)/temp)){score=s;if(s<bestscore){bestscore=s;best=cur;}}else std::swap(cur[i],alt[i]);
 }}std::copy(best.begin(),best.end(),a);return bestscore;
}
