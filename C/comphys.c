#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "comphys.h"
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define NR_END 1
#define FREE_ARG char*
#define NMAX 5000
#define ITMAX 20000
#define TOL 1.0e-2
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SWAP(a,b) {double temp=(a);(a)=(b);(b)=temp;}
#define GET_PSUM \
		   for(j=1;j<=ndim;j++) {\
		   for(sum=0.0,i=1;i<=mpts;i++) sum += p[i][j];\
		   psum[j] = sum;}

int ncom;
float *pcom,*xicom,(*nrfunc)(float []);

/* Note - most of what follows was taken from Numerical Recipies in
C, 2nd edition.  This is copyrighted material and should not be used without
acknowledgement, and should not be used commercially without permission. */

void nrerror(char error_text[])
{
  fprintf(stderr,"Computational Physics run-time error ...\n");
  fprintf(stderr,"%s\n",error_text);
  fprintf(stderr,"...now exiting to system...\n");
  exit(1);
}

float *vector(long nl, long nh)
{
  float *v;

  v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
  if(!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END;
}

double *dvector(long nl, long nh)
{
  double *v;

  v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
  if(!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END;
}

int *ivector(long nl, long nh)
{
  int *v;

  v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
  if(!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END;
}

float **matrix(long nrl, long nrh, long ncl, long nch)
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;

  m=(float **) malloc((size_t)((nrow+NR_END)*sizeof(float*)));
  if(!m) nrerror("allocation failrue 1 in matrix()");
  m += NR_END;
  m-= nrl;

  m[nrl]=(float *)malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
  if(!m[nrl]) nrerror("allocation failure 2 in matirx()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i] = m[i-1]+ncol;
  return m;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  double **m;

  m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
  if(!m) nrerror("allocation failrue 1 in matrix()");
  m += NR_END;
  m-= nrl;

  m[nrl]=(double *)malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
  if(!m[nrl]) nrerror("allocation failure 2 in matirx()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i] = m[i-1]+ncol;
  return m;
}

int **imatrix(long nrl, long nrh, long ncl, long nch)
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  int **m;

  m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
  if(!m) nrerror("allocation failrue 1 in matrix()");
  m += NR_END;
  m-= nrl;

  m[nrl]=(int *)malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
  if(!m[nrl]) nrerror("allocation failure 2 in matirx()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i] = m[i-1]+ncol;
  return m;
}

void free_vector(float *v, long nl, long nh)
{
  free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
{
  free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
{
  free((FREE_ARG) (v+nl-NR_END));
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
{
  free((FREE_ARG) (m[nrl]+ncl-NR_END));
  free((FREE_ARG) (m+nrl-NR_END));
}



/* Following are some C functions we will use in lab */


/* The amoeba function for use in the multidimensional downhill simplex
   algorithm */
void amoeba(float **p,float y[], int ndim, float ftol,
	    float (*funk)(float []), int *nfunk)
{
  float amotry(float **p, float y[], float psum[], int ndim,
	       float (*funk) (float []), int ihi, float fac);
  int i,ihi,ilo,inhi,j,mpts=ndim+1;
  float rtol,sum,swap,ysave,ytry,*psum;

  psum=vector(1,ndim);
  *nfunk=0;
  GET_PSUM
  for(;;) {
    ilo=1;
    ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
    for(i=1;i<=mpts;i++) {
      if(y[i] <= y[ilo]) ilo=i;
      if(y[i] > y[ihi]) {
	inhi=ihi;
	ihi=i;
      } else if (y[i] > y[inhi] && i != ihi) inhi=i;
    }
    rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo]));
    if(rtol < ftol) {
      SWAP(y[1],y[ilo])
      for(i=1;i<=ndim;i++) SWAP(p[1][i],p[ilo][i])
      break;
    }
    if(*nfunk >= NMAX) {
      printf("Too many iterations\n");
      return;
    }
    *nfunk += 2;
    ytry = amotry(p,y,psum,ndim,funk,ihi,-1.0);
    if(ytry <= y[ilo])
      ytry = amotry(p,y,psum,ndim,funk,ihi,2.0);
    else if (ytry >= y[inhi]) {
      ysave = y[ihi];
      ytry = amotry(p,y,psum,ndim,funk,ihi,0.5);
      if(ytry >= ysave) {
	for(i=1;i<=mpts;i++) {
	  if(i != ilo) {
	    for(j=1;j<=ndim;j++)
	      p[i][j] = psum[j] = 0.5*(p[i][j]+p[ilo][j]);
	    y[i] = (*funk)(psum);
	  }
	}
	*nfunk += ndim;
	GET_PSUM
      }
    } else --(*nfunk);
  }
  free_vector(psum,1,ndim);
}

float amotry(float **p, float y[], float psum[], int ndim,
	     float (*funk)(float []), int ihi, float fac)
{
  int j;
  float fac1,fac2,ytry,*ptry;

  ptry = vector(1,ndim);
  fac1 = (1.0-fac)/ndim;
  fac2 = fac1-fac;
  for(j=1;j<=ndim;j++) ptry[j] = psum[j]*fac1 - p[ihi][j]*fac2;
  if(ptry[3] < 0.0) ptry[3] = fabs(ptry[3]) + 0.5;
  ytry = (*funk)(ptry);
  if(ytry < y[ihi]) {
    y[ihi] = ytry;
    for(j=1;j<=ndim;j++) {
      psum[j] += ptry[j]-p[ihi][j];
      p[ihi][j] = ptry[j];
    }
  }
  free_vector(ptry,1,ndim);
  return ytry;
}


/* The blackbody Planck function */
float planck(float wave,float T)
{
  static float p = 1.19106e+27;
  float p1;
  p1 = pow(wave,5.0)*(exp(1.43879e+08/(wave*T)) -1.0);
  return(p/p1);
}


/* Bessel function of the 0th order */
float bessj0(float x)
{
   float ax,z;
   double xx,y,ans,ans1,ans2;

   if((ax=fabs(x)) < 8.0) {
     y=x*x;
     ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
          +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
     ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
          +y*(59272.64853+y*(267.8532712+y*1.0))));
     ans = ans1/ans2;
   } else {
     z=8.0/ax;
     y=z*z;
     xx=ax-0.785398164;
     ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
         +y*(-0.2073370639e-5+y*0.2093887211e-6)));
     ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         -y*0.934935152e-7)));
     ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
   }
   return ans;
}

/* Bessel function of the 1st order */
float bessj1(float x)
{
  float ax,z;
  double xx,y,ans,ans1,ans2;

  if((ax=fabs(x)) < 8.0) {
    y=x*x;
    ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
	+y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
    ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
	+y*(99447.43394+y*(376.9991397+y*1.0))));
    ans=ans1/ans2;
  } else {
    z=8.0/ax;
    y=z*z;
    xx=ax-2.356194491;
    ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
	+y*(0.2457520174e-5+y*(-0.240337019e-6))));
    ans2=0.04687499995+y*(-0.2002690873e-3
	+y*(0.8449199096e-5+y*(-0.88228987e-6
	+y*0.105787412e-6)));
    ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    if(x < 0.0) ans = -ans;
  }
  return ans;
}

float ran1(long *idum)
{
  int j;
  long k;
  static long iy=0;
  static long iv[NTAB];
  float temp;

  if(*idum <= 0 || !iy) {
    if(-(*idum) < 1) *idum=1;
    else *idum = -(*idum);
    for(j=NTAB+7;j>=0;j--) {
      k=(*idum)/IQ;
      *idum=IA*(*idum-k*IQ)-IR*k;
      if(*idum < 0) *idum += IM;
      if(j < NTAB) iv[j] = *idum;
    }
    iy=iv[0];
  }
  k=(*idum)/IQ;
  *idum=IA*(*idum-k*IQ)-IR*k;
  if(*idum < 0) *idum += IM;
  j=iy/NDIV;
  iy=iv[j];
  iv[j] = *idum;
  if((temp=AM*iy) > RNMX) return RNMX;
  else return temp;
}

float gasdev(long *idum)
{
  float ran1(long *idum);
  static int iset=0;
  static float gset;
  float fac,rsq,v1,v2;

  if(iset == 0) {
    do {
      v1=2.0*ran1(idum)-1.0;
      v2=2.0*ran1(idum)-1.0;
      rsq=v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac=sqrt(-2.0*log(rsq)/rsq);
    gset=v1*fac;
    iset=1;
    return v2*fac;
  } else {
    iset=0;
    return gset;
  }
}

void powell(float p[],float **xi,int n,float ftol,int *iter,float *fret,
   float (*func)(float []))
{
  void linmin(float p[], float xi[], int n, float *fret, 
       float (*func)(float []));
  int i,ibig,j;
  float del,fp,fptt,t,*pt,*ptt,*xit;

  pt = vector(1,n);
  ptt = vector(1,n);
  xit = vector(1,n);
  *fret = (*func)(p);
  for(j=1;j<=n;j++) pt[j] = p[j];
  for(*iter-1;;++(*iter)) {
    fp = (*fret);
    ibig = 0;
    del = 0.0;
    for(i=1;i<=n;i++) {
      for(j=1;j<=n;j++) xit[j] = xi[j][i];
      fptt = (*fret);
      linmin(p,xit,n,fret,func);
      if(fabs(fptt-(*fret)) > del) {
        del = fabs(fptt-(*fret));
        ibig = i;
      }
    }
    if(2.0*fabs(fp-(*fret)) <= ftol*(fabs(fp)+fabs(*fret))) {
      free_vector(xit,1,n);
      free_vector(ptt,1,n);
      free_vector(pt,1,n);
      return;
    }
    if(*iter == ITMAX) nrerror("powell exceeding maximum interations.");
    for(j=1;j<=n;j++) {
      ptt[j] = 2.0*p[j]-pt[j];
      xit[j] = p[j]-pt[j];
      pt[j] = p[j];
    }
    fptt = (*func)(ptt);
    if(fptt < fp) {
      t = 2.0*(fp-2.0*(*fret)+fptt)*SQR(fp-(*fret)-del)-del*SQR(fp-fptt);
      if(t < 0.0) {
        linmin(p,xit,n,fret,func);
        for(j=1;j<=n;j++) {
          xi[j][ibig] = xi[j][n];
          xi[j][n] = xit[j];
        }
      }
    }
  }
}


void linmin(float p[], float xi[], int n, float *fret, 
           float (*func)(float []))
{
  float brent(float ax, float bx, float cx, float (*f)(float), float tol,
              float *xmin);
  float f1dim(float x);
  void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,
              float *fc, float (*func)(float));
  int j;
  float xx,xmin,fx,fb,fa,bx,ax;

  ncom = n;
  pcom = vector(1,n);
  xicom = vector(1,n);
  nrfunc = func;
  for(j=1;j<=n;j++) {
    pcom[j] = p[j];
    xicom[j] = xi[j];
  }
  ax = 0.0;
  xx = 1.0;
  mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
  *fret = brent(ax,xx,bx,f1dim,TOL,&xmin);
  for(j=1;j<=n;j++) {
    xi[j] *= xmin;
    p[j] += xi[j];
  }
  free_vector(xicom,1,n);
  free_vector(pcom,1,n);
}

float f1dim(float x)
{
  int j;
  float f,*xt;

  xt = vector(1,ncom);
  for(j=1;j<=ncom;j++) xt[j] = pcom[j] + x*xicom[j];
  f = (*nrfunc)(xt);
  free_vector(xt,1,ncom);
  return f;
}

float brent(float ax, float bx, float cx, float (*f)(float), float tol,
            float *xmin)
{
  int iter;
  float a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
  float e = 0.0;
  
  a = (ax < cx ? ax : cx);
  b = (ax > cx ? ax : cx);
  x = w = v = bx;
  fw = fv = fx = (*f)(x);
  for(iter=1;iter<=ITMAX;iter++) {
    xm = 0.5*(a+b);
    tol2 = 2.0*(tol1 = tol*fabs(x)+ZEPS);
    if(fabs(x-xm) <= (tol2 - 0.5*(b-a))) {
      *xmin = x;
      return fx;
    }
    if(fabs(e) > tol1) {
      r = (x-w)*(fx-fv);
      q = (x-v)*(fx-fw);
      p = (x-v)*q - (x-w)*r;
      q = 2.0*(q-r);
      if(q > 0.0) p = -p;
      q = fabs(q);
      etemp = e;
      e = d;
      if(fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
        d = CGOLD*(e=(x >= xm ? a-x : b-x));
      else {
        d = p/q;
        u = x + d;
        if(u-a < tol2 || b-u < tol2) d = SIGN(tol1,xm-x);
      }
   } else {
      d = CGOLD*(e=(x >= xm ? a-x : b-x));
   }
   u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
   fu = (*f)(u);
   if(fu <= fx) {
     if(u >= x) a=x; else b=x;
     SHFT(v,w,x,u)
     SHFT(fv,fw,fx,fu)
   } else {
     if(u < x) a=u; else b=u;
     if(fu <= fw || w == x) {
       v = w;
       w = u;
       fv = fw;
       fw = fu;
     } else if (fu <= fv || v == x || v == w) {
       v = u;
       fv = fu;
     }
   }
 }
 nrerror("Too many iterations in brent");
}

void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb, 
float *fc, float (*func)(float))
{
  float ulim,u,r,q,fu,dum;

  *fa = (*func)(*ax);
  *fb = (*func)(*bx);
  if(*fb > *fa) {
    SHFT(dum,*ax,*bx,dum)
    SHFT(dum,*fb,*fa,dum)
  }
  *cx = (*bx) + GOLD*(*bx - *ax);
  *fc = (*func)(*cx);
  while(*fb > *fc) {
    r = (*bx - *ax)*(*fb - *fc);
    q = (*bx - *cx)*(*fb - *fa);
    u = (*bx) - ((*bx - *cx)*q - (*bx - *ax)*r)/
        (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
    ulim = (*bx) + GLIMIT*(*cx - *bx);
    if((*bx-u)*(u- *cx) > 0.0) {
      fu = (*func)(u);
      if(fu < *fc) {
        *ax = (*bx);
        *bx = u;
        *fa = (*fb);
        *fb = fu;
        return;
      } else if(fu > *fb) {
        *cx = u;
        *fc = fu;
        return;
      }
      u = (*cx) + GOLD*(*cx - *bx);
      fu = (*func)(u);
    } else if ((*cx - u)*(u - ulim) > 0.0) {
      fu = (*func)(u);
      if(fu < *fc) {
        SHFT(*bx,*cx,u,*cx+GOLD*(*cx - *bx))
        SHFT(*fb,*fc,fu,(*func)(u))
      }
    } else if((u-ulim)*(ulim- *cx) >= 0.0) {
      u = ulim;
      fu = (*func)(u);
    } else {
      u = (*cx)+GOLD*(*cx - *bx);
      fu = (*func)(u);
    }
    SHFT(*ax,*bx,*cx,u)
    SHFT(*fa,*fb,*fc,fu)
  }
}

void covsrt(float **covar, int ma,int ia[], int mfit)
{
  int i,j,k;
  float swap;

  for(i=mfit+1;i<=ma;i++)
     for(j=1;j<=i;j++) covar[i][j]=covar[j][i]=0.0;
  k=mfit;
  for(j=ma;j>=1;j--) {
    if(ia[j]) {
      for(i=1;i<=ma;i++) SWAP(covar[i][k],covar[i][j])
      for(i=1;i<=ma;i++) SWAP(covar[k][i],covar[j][i])
      k--;
    }
  }
}


void gaussj(float **a,int n,float **b,int m)
{
   int *indxc,*indxr,*ipiv;
   int i,icol,irow,j,k,l,ll;
   float big,dum,pivinv;

   indxc = ivector(1,n);
   indxr = ivector(1,n);
   ipiv = ivector(1,n);
   for(j=1;j<=n;j++) ipiv[j] = 0;
   for(i=1;i<=n;i++) {
      big = 0.0;
      for(j=1;j<=n;j++)
	 if(ipiv[j] != 1)
	    for(k=1;k<=n;k++) {
	       if(ipiv[k] == 0) {
		 if(fabs(a[j][k]) >= big) {
		   big = fabs(a[j][k]);
		   irow = j;
		   icol = k;
		 }
	       } else if(ipiv[k] > 1) nrerror("GAUSSJ: Singular Matrix-1");
	    }
      ++(ipiv[icol]);
      if(irow != icol) {
	for(l=1;l<=n;l++) SWAP(a[irow][l],a[icol][l])
	for(l=1;l<=m;l++) SWAP(b[irow][l],b[icol][l])
      }
      indxr[i] = irow;
      indxc[i] = icol;
      if(a[icol][icol] == 0.0) nrerror("GAUSSJ: Singular Matrix-2");
      pivinv = 1.0/a[icol][icol];
      a[icol][icol] = 1.0;
      for(l=1;l<=n;l++) a[icol][l] *= pivinv;
      for(l=1;l<=m;l++) b[icol][l] *= pivinv;
      for(ll=1;ll<=n;ll++)
	 if(ll != icol) {
	   dum = a[ll][icol];
	   a[ll][icol] = 0.0;
	   for(l=1;l<=n;l++) a[ll][l] -= a[icol][l]*dum;
	   for(l=1;l<=m;l++) b[ll][l] -= b[icol][l]*dum;
	 }
   }
   for(l=n;l>=1;l--) {
      if(indxr[l] != indxc[l])
	for(k=1;k<=n;k++)
	   SWAP(a[k][indxr[l]],a[k][indxc[l]]);
   }
   free_ivector(ipiv,1,n);
   free_ivector(indxr,1,n);
   free_ivector(indxc,1,n);
}

void mrqmin(float x[],float y[],float sig[],int ndata,float a[],int ia[],
            int ma,float **covar,float **alpha,float *chisq,
	    void (*funcs)(float, float [],float *,float [], int), 
	    float *alamda)
{
  void covsrt(float **covar, int ma, int ia[], int mfit);
  void gaussj(float **a, int n, float **b, int m);
  void mrqcof(float x[],float y[],float sig[],int ndata,float a[],int ia[],
	    int ma,float **alpha,float beta[],float *chisq,
	      void (*funcs)(float,float [],float *,float [],int));
  int j,k,l;
  static int mfit;
  static float ochisq,*atry,*beta,*da,**oneda;

  if(*alamda < 0.0) {
    atry = vector(1,ma);
    da = vector(1,ma);
    beta = vector(1,ma);
    for(mfit=0,j=1;j<=ma;j++) 
      if(ia[j]) mfit++;
    oneda=matrix(1,mfit,1,1);
    *alamda=0.001;
    mrqcof(x,y,sig,ndata,a,ia,ma,alpha,beta,chisq,funcs);
    ochisq = (*chisq);
    for(j=1;j<=ma;j++) atry[j]=a[j];
  }
  for(j=1;j<=mfit;j++) {
    for(k=1;k<=mfit;k++) covar[j][k] = alpha[j][k];
    covar[j][j] = alpha[j][j]*(1.0+(*alamda));
    oneda[j][1] = beta[j];
  }
  gaussj(covar,mfit,oneda,1);
  for(j=1;j<=mfit;j++) da[j] = oneda[j][1];
  if(*alamda == 0.0) {
    covsrt(covar,ma,ia,mfit);  
    free_vector(beta,1,ma);
    free_vector(da,1,ma);
    free_vector(atry,1,ma);
    free_matrix(oneda,1,mfit,1,1);
    return;
  }
  for(j=0,l=1;l<=ma;l++) 
    if(ia[l]) atry[l] = a[l]+da[++j];
  mrqcof(x,y,sig,ndata,atry,ia,ma,covar,da,chisq,funcs);
  if(*chisq < ochisq) {
    *alamda *= 0.1;
    ochisq = (*chisq);
    for(j=1;j<=mfit;j++) {
      for(k=1;k<=mfit;k++) alpha[j][k] = covar[j][k];
      beta[j] = da[j];
    }
    for(l=1;l<=ma;l++) a[l]=atry[l];
  } else {
    *alamda *= 10.0;
    *chisq = ochisq;
  }
}

void mrqcof(float x[],float y[],float sig[],int ndata,float a[],int ia[],
	    int ma,float **alpha,float beta[],float *chisq,
            void (*funcs)(float,float [],float *,float [],int))
{
  int k,j,i,l,m,mfit=0;
  float ymod,wt,sig2i,dy,*dyda;

  dyda = vector(1,ma);
  for(j=1;j<=ma;j++) 
    if(ia[j]) mfit++;
  for(j=1;j<=mfit;j++) {
    for(k=1;k<=j;k++) alpha[j][k] = 0.0;
    beta[j] = 0.0;
  }
  *chisq=0.0;
  for(i=1;i<=ndata;i++) {
    (*funcs)(x[i],a,&ymod,dyda,ma);
    sig2i = 1.0/(sig[i]*sig[i]);
    dy = y[i] - ymod;
    for(j=0,l=1;l<=ma;l++) {
      if(ia[l]) {
        wt = dyda[l]*sig2i;
        for(j++,k=0,m=1;m<=l;m++)
	  if(ia[m]) alpha[j][++k] += wt*dyda[m];
        beta[j] += dy*wt;
      }
    }
    *chisq += dy*dy*sig2i;
  }
  for(j=2;j<=mfit;j++)
    for(k=1;k<j;k++) alpha[k][j] = alpha[j][k];
  free_vector(dyda,1,ma);
}


void lfit(float x[],float y[],float sig[],int ndat,float a[],int ia[],
	  int ma,float **covar,float *chisq, void (*funcs)(float,float [],int))
{
  void covsrt(float **covar,int ma,int ia[],int mfit);
  void gaussj(float **a,int n,float **b,int m);
  int i,j,k,l,m,mfit=0;
  float ym,wt,sum,sig2i,**beta,*afunc;

  beta = matrix(1,ma,1,1);
  afunc = vector(1,ma);
  for(j=1;j<=ma;j++)
    if(ia[j]) mfit++;
  if(mfit == 0) nrerror("lfit: no parameters to be fitted");
  for(j=1;j<=mfit;j++) {
    for(k=1;k<=mfit;k++) covar[j][k] = 0.0;
    beta[j][1] = 0.0;
  }
  for(i=1;i<=ndat;i++) {
    (*funcs)(x[i],afunc,ma);
    ym = y[i];
    if(mfit < ma) {
      for(j=1;j<=ma;j++)
	if(!ia[j]) ym -= a[j]*afunc[j];
    }
    sig2i = 1.0/SQR(sig[i]);
    for(j=0,l=1;l<=ma;l++) {
      if(ia[l]) {
	wt = afunc[l]*sig2i;
	for(j++,k=0,m=1;m<=l;m++)
	  if(ia[m]) covar[j][++k] += wt*afunc[m];
	beta[j][1] += ym*wt;
      }
    }
  }
  for(j=2;j<=mfit;j++)
    for(k=1;k<j;k++)
      covar[k][j] = covar[j][k];
  gaussj(covar,mfit,beta,1);
  for(j=0,l=1;l<=ma;l++)
    if(ia[l]) a[l] = beta[++j][1];
  *chisq = 0.0;
  for(i=1;i<=ndat;i++) {
    (*funcs)(x[i],afunc,ma);
    for(sum=0.0,j=1;j<=ma;j++) sum += a[j]*afunc[j];
    *chisq += SQR((y[i]-sum)/sig[i]);
  }
  covsrt(covar,ma,ia,mfit);
  free_vector(afunc,1,ma); 
  free_matrix(beta,1,ma,1,1); 
}


void four1(float data[],unsigned long nn,int isign)
{
  unsigned long  n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  float tempr,tempi;

  n = nn << 1;
  j = 1;
  for(i=1;i<n;i+=2) {
    if(j>i) {
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m = nn;
    while(m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax = 2;
  while(n > mmax) {
    istep = mmax << 1;
    theta = isign*(6.28318530717959/mmax);
    wtemp = sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi = sin(theta);
    wr = 1.0;
    wi = 0.0;
    for(m=1;m<mmax;m+=2) {
      for(i=m;i<=n;i+=istep) {
	j = i+mmax;
	tempr = wr*data[j]-wi*data[j+1];
	tempi = wr*data[j+1]+wi*data[j];
	data[j] = data[i]-tempr;
	data[j+1]=data[i+1]-tempi;
	data[i] += tempr;
	data[i+1] += tempi;
      }
      wr = (wtemp=wr)*wpr - wi*wpi+wr;
      wi = wi*wpr + wtemp*wpi + wi;
    }
    mmax = istep;
  }
}


void realft(float data[], unsigned long n, int isign)
{
   void four1(float data[], unsigned long nn, int isign);
   unsigned long i,i1,i2,i3,i4,np3;
   float c1=0.5,c2,h1r,h1i,h2r,h2i;
   double wr,wi,wpr,wpi,wtemp,theta;

   theta=3.141592653589793/(double) (n>>1);
   if (isign == 1) {
      c2 = -0.5;
      four1(data,n>>1,1);
   } else {
      c2=0.5;
      theta = -theta;
   }
   wtemp=sin(0.5*theta);
   wpr = -2.0*wtemp*wtemp;
   wpi=sin(theta);
   wr=1.0+wpr;
   wi=wpi;
   np3=n+3;
   for (i=2;i<=(n>>2);i++) {
      i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
      h1r=c1*(data[i1]+data[i3]);
      h1i=c1*(data[i2]-data[i4]);
      h2r = -c2*(data[i2]+data[i4]);
      h2i=c2*(data[i1]-data[i3]);
      data[i1] = h1r+wr*h2r-wi*h2i;
      data[i2] = h1i+wr*h2i+wi*h2r;
      data[i3] = h1r-wr*h2r+wi*h2i;
      data[i4] = -h1i+wr*h2i+wi*h2r;
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
   }
   if (isign == 1) {
      data[1] = (h1r=data[1])+data[2];
      data[2] = h1r-data[2];
   } else {
      data[1]=c1*((h1r=data[1])+data[2]);
      data[2]=c1*(h1r-data[2]);
      four1(data,n>>1,-1);
   }
}


void twofft(float data1[], float data2[], float fft1[], float fft2[],
      unsigned long n)
/* Given two real input arrays data1[1..n] and data2[1..n], this routine
calls four1 and returns two complex output arrays, fft1[1..2n] and
fft2[1..2n], each of complex length n (i.e., real length 2*n), which
contain the discrete Fourier transforms of the respective data arrays. n
MUST be an integer power of 2. */
{
   void four1(float data[], unsigned long nn, int isign);
   unsigned long nn3,nn2,jj,j;
   float rep,rem,aip,aim;
   nn3=1+(nn2=2+n+n);
   for (j=1,jj=2;j<=n;j++,jj+=2) { 
   /* Pack the two real arrays into one complex array. */
      fft1[jj-1]=data1[j];
      fft1[jj]=data2[j];
   }
   four1(fft1,n,1); /* Transform the complex array. */
   fft2[1]=fft1[2];
   fft1[2]=fft2[2]=0.0;
   for (j=3;j<=n+1;j+=2) {
      /* Use symmetries to separate the two transforms. */
      rep=0.5*(fft1[j]+fft1[nn2-j]); 
      rem=0.5*(fft1[j]-fft1[nn2-j]);
      aip=0.5*(fft1[j+1]+fft1[nn3-j]);
      aim=0.5*(fft1[j+1]-fft1[nn3-j]);
      fft1[j]=rep; /* Ship them out in two complex arrays. */
      fft1[j+1]=aim;
      fft1[nn2-j]=rep;
      fft1[nn3-j] = -aim;
      fft2[j]=aip;
      fft2[j+1] = -rem;
      fft2[nn2-j]=aip;
      fft2[nn3-j]=rem;
   }
}

void convlv(float data[], unsigned long n, float respns[], unsigned long m,
            int isign, float ans[])
/* Convolves or deconvolves a real data set data[1..n] (including any
user-supplied zero padding) with a response function respns[1..n]. The
response function must be stored in wrap-around order in the first m
elements of respns, where m is an odd integer.  Wrap-around order
means that the first half of the array respns contains the impulse
response function at positive times, while the second half of the array
contains the impulse response function at negative times, counting down
from the highest element respns[m]. On input isign is +1 for
convolution, n. The answer is returned in the first n components of ans.
However, ans must be supplied in the calling program with dimensions
[1..2*n], for consistency with twofft. n MUST be an integer power of
two. */
{
   void realft(float data[], unsigned long n, int isign);
   void twofft(float data1[], float data2[], float fft1[], float fft2[],
        unsigned long n);
   unsigned long i,no2;
   float dum,mag2,*fft;

   fft=vector(1,n<<1);
   for (i=1;i<=(m-1)/2;i++) /* Put respns in array of length n. */
      respns[n+1-i]=respns[m+1-i];
   for (i=(m+3)/2;i<=n-(m-1)/2;i++) /* Pad with zeros. */
      respns[i]=0.0;
   twofft(data,respns,fft,ans,n); /* FFT both at once. */
   no2=n>>1;
   for (i=2;i<=n+2;i+=2) {
      if (isign == 1) {
	 /* Multiply FFTs to convolve. */
         ans[i-1]=(fft[i-1]*(dum=ans[i-1])-fft[i]*ans[i])/no2; 
         ans[i]=(fft[i]*dum+fft[i-1]*ans[i])/no2;
      } else if (isign == -1) {
         if ((mag2=SQR(ans[i-1])+SQR(ans[i])) == 0.0)
            nrerror("Deconvolving at response zero in convlv");
	 /* Divide FFTs to deconvolve. */ 
         ans[i-1]=(fft[i-1]*(dum=ans[i-1])+fft[i]*ans[i])/mag2/no2; 
	 ans[i]=(fft[i]*dum-fft[i-1]*ans[i])/mag2/no2;
      } else nrerror("No meaning for isign in convlv");
   }
   ans[2]=ans[n+1]; /* Pack last element with first for realft. */
   realft(ans,n,-1); /* Inverse transform back to time domain. */
   free_vector(fft,1,n<<1);
}


void correl(float data1[], float data2[], unsigned long n, float ans[])
{
  void realft(float data[], unsigned long n, int isign);
  void twofft(float data1[], float data2[], float fft1[], float fft2[],
        unsigned long n);
  unsigned long no2,i;
  float dum,*fft;

  fft = vector(1,n<<1);
  twofft(data1,data2,fft,ans,n);
  no2=n>>1;
  for(i=2;i<=n+2;i+=2) {
    ans[i-1]=(fft[i-1]*(dum=ans[i-1])+fft[i]*ans[i])/no2;
    ans[i] = (fft[i]*dum-fft[i-1]*ans[i])/no2;
  }
  ans[2] = ans[n+1];
  realft(ans,n,-1);
  free_vector(fft,1,n<<1);
}

void qsfloat( float a[], int l, int r)
{
   int partfloat( float[], int, int);   
   int j;
   if( l < r ) 
   {
   	// divide and conquer
        j = partfloat( a, l, r);
        qsfloat( a, l, j-1);
        qsfloat( a, j+1, r);
   }	
}

int partfloat( float a[], int l, int r) {
   int i, j;
   float pivot, t;
   pivot = a[l];
   i = l; j = r+1;	
   while( 1)
   {
   	do ++i; while( a[i] <= pivot && i <= r );
   	do --j; while( a[j] > pivot );
   	if( i >= j ) break;
   	t = a[i]; a[i] = a[j]; a[j] = t;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   return j;
}

//qsint.c
void qsint( int a[], int l, int r)
{
   int partint( int[], int, int);
   int j;
   if( l < r ) 
   {
   	// divide and conquer
        j = partint( a, l, r);
        qsint( a, l, j-1);
        qsint( a, j+1, r);
   }	
}

int partint( int a[], int l, int r) {
   int i, j;
   int pivot, t;
   pivot = a[l];
   i = l; j = r+1;	
   while( 1)
   {
   	do ++i; while( a[i] <= pivot && i <= r );
   	do --j; while( a[j] > pivot );
   	if( i >= j ) break;
   	t = a[i]; a[i] = a[j]; a[j] = t;
   }
   t = a[l]; a[l] = a[j]; a[j] = t;
   return j;
}
