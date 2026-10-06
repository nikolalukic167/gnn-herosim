#include "mixed.cpp"

static double refine_pair(const Mixed& b, std::vector<int64_t>& a, int rounds) {
 const int n=b.b.J*b.b.O,h=b.b.H;
 double cost=mixed_replay(b,a.data());
 for(int r=0;r<rounds;++r){
  double best=cost;int chosen=-1,host=-1;
  for(int i=0;i<n;++i)for(int v=0;v<h;++v)if(v!=a[i]&&std::isfinite(b.b.p[i*h+v])){
   int old=a[i];a[i]=v;double c=mixed_replay(b,a.data());a[i]=old;
   if(c<best){best=c;chosen=i;host=v;}
  }
  if(chosen<0)break;
  a[chosen]=host;cost=best;
 }
 return cost;
}
extern "C" void mixed_pairs(const double*q,const int64_t*start,int j,int o,int h,int g,int k,
 const int64_t*pairs,int count,int rounds,double*costs,int64_t*plans){
 Mixed b(Problem(q,j,o,h,g,k,7));const int n=j*o;
 for(int row=0;row<count;++row){
  std::vector<int64_t>a(start,start+n);
  for(int z=0;z<2;++z){int i=pairs[2*row+z];for(int v=0;v<h;++v)if(v!=start[i]&&std::isfinite(b.b.p[i*h+v])){a[i]=v;break;}}
  costs[row]=refine_pair(b,a,rounds);std::copy(a.begin(),a.end(),plans+row*n);
 }
}
