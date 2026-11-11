#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define next sm_next
#include "splitmix64.c"
#undef next
#define next xo_next
#define s xo_s
#include "xo_trim.c"
#undef s
#undef next
static void seed(uint64_t v){ x = v; for(int i=0;i<4;i++) xo_s[i]=sm_next(); }
static double u(void){ return (double)(xo_next()>>11) * 0x1.0p-53; }
static uint64_t below(uint64_t n){ uint64_t th=(0-n)%n; for(;;){uint64_t r=xo_next(); if(r>=th) return r%n;} }
static double normal(void){ double a=0.0; for(int i=0;i<12;i++) a = a + u(); return a-6.0; }
static uint64_t bits(double d){ uint64_t b; memcpy(&b,&d,8); return b; }
static long long prob_round(double v){ double f=floor(v); return (long long)f + (u() < (v-f) ? 1:0); }
static int weighted(const double*w,int n){ double t=0.0; for(int i=0;i<n;i++) t=t+w[i]; double xx=u()*t; double acc=0.0;
  for(int i=0;i<n;i++){acc=acc+w[i]; if(xx<acc) return i;} for(int i=n-1;i>=0;i--) if(w[i]>0) return i; return -1; }
int main(void){
  uint64_t seeds[]={0,42,0xFFFFFFFFFFFFFFFFull};
  uint64_t ns[]={1,2,3,10,1000,0x8000000000000001ull,0xFFFFFFFFFFFFFFFFull};
  double pr[]={0.0,0.5,2.25,7.0,0.999};
  double w[]={40,30,20,10};
  for(int si=0;si<3;si++){
    uint64_t sd=seeds[si];
    printf("{\"kind\":\"rng\",\"seed\":%llu,\"next_u64\":[", (unsigned long long)sd);
    seed(sd); for(int i=0;i<20;i++) printf("%s\"%016llx\"", i?",":"", (unsigned long long)xo_next());
    printf("],\"u\":["); seed(sd); for(int i=0;i<10;i++) printf("%s\"%016llx\"", i?",":"", (unsigned long long)bits(u()));
    printf("],\"below\":[");
    for(int ni=0;ni<7;ni++){ seed(sd); printf("%s{\"n\":%llu,\"values\":[", ni?",":"", (unsigned long long)ns[ni]);
      for(int i=0;i<10;i++) printf("%s%llu", i?",":"", (unsigned long long)below(ns[ni])); printf("]}"); }
    printf("],\"normal\":["); seed(sd); for(int i=0;i<10;i++) printf("%s\"%016llx\"", i?",":"", (unsigned long long)bits(normal()));
    printf("],\"prob_round\":[");
    for(int pi=0;pi<5;pi++){ seed(sd); printf("%s{\"x\":\"%016llx\",\"values\":[", pi?",":"", (unsigned long long)bits(pr[pi]));
      for(int i=0;i<10;i++) printf("%s%lld", i?",":"", prob_round(pr[pi])); printf("]}"); }
    printf("],\"weighted\":[{\"w\":[\"%016llx\",\"%016llx\",\"%016llx\",\"%016llx\"],\"values\":[", (unsigned long long)bits(w[0]),(unsigned long long)bits(w[1]),(unsigned long long)bits(w[2]),(unsigned long long)bits(w[3]));
    seed(sd); for(int i=0;i<20;i++) printf("%s%d", i?",":"", weighted(w,4));
    printf("]}]}\n");
  }
}
