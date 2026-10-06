#include "dispatch.cpp"
extern "C" double adaptive_priority(const double*q,const int64_t*a,int64_t*r,int j,int o,int h,int g,int k,int mode){
 Mixed m(Problem(q,j,o,h,g,k,7));const auto&b=m.b;
 int next[64]={},running[32],last[32];double finish[32]={},readytime[64]={};
 std::fill(running,running+h,-1);std::fill(last,last+h,-1);
 std::vector<double>tail(j*o),pressure(j*o);int freq[30]={};
 for(int i=0;i<j*o;++i)for(int bit=0;bit<30;++bit)if(m.locks[i]&(1<<bit))++freq[bit];
 for(int job=0;job<j;++job)for(int op=o-1;op>=0;--op){int i=job*o+op;tail[i]=b.p[i*h+a[i]]+(op+1<o?tail[i+1]:0.);double v=0;for(int bit=0;bit<30;++bit)if(m.locks[i]&(1<<bit))v+=freq[bit];pressure[i]=v+(op+1<o?pressure[i+1]:0.);}
 uint64_t busy=0;uint32_t hosts=0;int held=0,done=0,starts=0;double t=0,total=0;
 while(done<j*o){
  for(int host=0;host<h;++host)if(running[host]>=0&&finish[host]<=t+1e-9){int i=running[host],job=i/o;running[host]=-1;last[host]=m.types[i];++next[job];++done;readytime[job]=t;busy&=~(uint64_t(1)<<job);hosts&=~(uint32_t(1)<<host);held&=~m.locks[i];if(next[job]==o)total+=t;}
  if(done==j*o)break;
  std::vector<int>ready;double key[64]={};
  for(int job=0;job<j;++job)if(!(busy&(uint64_t(1)<<job))&&next[job]<o)ready.push_back(job);
  for(int job:ready){int i=job*o+next[job],host=a[i];double base=b.p[i*h+host],setup=last[host]>=0?b.setup[last[host]*k+m.types[i]]:0.;
   if(mode==0)key[job]=base+setup;
   else if(mode==1)key[job]=tail[i]+setup;
   else if(mode==2)key[job]=tail[i]+2*setup;
   else if(mode==3)key[job]=tail[i]+.5*setup;
   else if(mode==4)key[job]=-tail[i]+setup;
   else if(mode==5){int overlap=0;for(int other:ready)if(other!=job&&(m.locks[i]&m.locks[other*o+next[other]]))++overlap;key[job]=(tail[i]+setup)/(1+overlap);}
   else if(mode==6)key[job]=10000*setup+tail[i];
   else if(mode==7)key[job]=(tail[i]+setup)/std::max(1.,pressure[i]);
   else key[job]=readytime[job];
  }
  std::sort(ready.begin(),ready.end(),[&](int x,int y){return key[x]==key[y]?x<y:key[x]<key[y];});
  for(int job:ready){int i=job*o+next[job],host=a[i];if(running[host]>=0||(held&m.locks[i]))continue;
   bool blocked=false;for(int domain=0;domain<g;++domain)if((m.domains[domain]&(uint32_t(1)<<host))&&__builtin_popcount(m.domains[domain]&hosts)>=2){blocked=true;break;}if(blocked)continue;
   running[host]=i;finish[host]=t+b.p[i*h+host]+(last[host]>=0?b.setup[last[host]*k+m.types[i]]:0.);busy|=uint64_t(1)<<job;hosts|=uint32_t(1)<<host;held|=m.locks[i];r[i]=starts++;
  }
  double nt=std::numeric_limits<double>::infinity();for(int host=0;host<h;++host)if(running[host]>=0)nt=std::min(nt,finish[host]);if(!std::isfinite(nt))return nt;t=nt;
 }
 return total;
}
