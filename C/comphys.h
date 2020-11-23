static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
static float maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
(maxarg1) : (maxarg2))
void nrerror(char error_text[]);
float *vector(long nl, long nh);
int *ivector(long nl, long nh);
double *dvector(long nl, long nh);
float **matrix(long nrl, long nrh, long ncl, long nch);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
int **imatrix(long nrl, long nrh, long ncl, long nch);
void free_vector(float *v,long nl, long nh);
void free_ivector(int *v,long nl, long nh);
void free_dvector(double *v,long nl, long nh);
void free_matrix(float **m,long nrl,long nrh, long ncl, long nch);
void free_imatrix(int **m,long nrl,long nrh, long ncl, long nch);
void free_dmatrix(double **m,long nrl,long nrh, long ncl, long nch);

void amoeba(float **p,float y[], int ndim, float ftol,
	    float (*funk)(float []), int *nfunk);
float planck(float wave,float T);
float bessj0(float x);
float bessj1(float x);
float ran1(long *idum);
float gasdev(long *idum);
void powell(float p[],float **xi,int n,float ftol,int *iter,float *fret,
	    float (*func)(float []));
void linmin(float p[], float xi[], int n, float *fret, 
	    float (*func)(float []));
float brent(float ax, float bx, float cx, float (*f)(float), float tol,
              float *xmin);
float f1dim(float x);
void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,
              float *fc, float (*func)(float));
void covsrt(float **covar, int ma,int ia[], int mfit);
void gaussj(float **a,int n,float **b,int m);
void mrqmin(float x[],float y[],float sig[],int ndata,float a[],int ia[],
            int ma,float **covar,float **alpha,float *chisq,
	    void (*funcs)(float, float [],float *,float [], int), 
	    float *alamda);
void mrqcof(float x[],float y[],float sig[],int ndata,float a[],int ia[],
	    int ma,float **alpha,float beta[],float *chisq,
            void (*funcs)(float,float [],float *,float [],int));
void lfit(float x[],float y[],float sig[],int ndat,float a[],int ia[],
	  int ma,float **covar,float *chisq, 
          void (*funcs)(float,float [],int));
void four1(float data[],unsigned long nn,int isign);
void realft(float data[], unsigned long n, int isign);
void twofft(float data1[], float data2[], float fft1[], float fft2[],
	    unsigned long n);
void convlv(float data[], unsigned long n, float respns[], unsigned long m,
            int isign, float ans[]);
void correl(float data1[], float data2[], unsigned long n, float ans[]);
void qsint( int a[], int l, int r);
void qsfloat( float a[], int l, int r);
