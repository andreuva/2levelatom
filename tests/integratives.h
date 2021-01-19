/*******************************************************************************
Gauss-Legendre integration function, gauleg, from "Numerical Recipes in C"
(Cambridge Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and
B.P. Flannery
*******************************************************************************/
#include <stdlib.h>
#include <math.h>

#define EPS 3.0e-16 /* EPS is the relative precision. */
void gauleg(double x1, double x2, double x[], double w[], int n)
/*******************************************************************************
Given the lower and upper limits of integration x1 and x2, and given n, this
routine returns arrays x[1..n] and w[1..n] of length n, containing the abscissas
and weights of the Gauss-Legendre n-point quadrature formula.
*******************************************************************************/
{
	int m,j,i;
	double z1,z,xm,xl,pp,p3,p2,p1;
	m=(n+1)/2; /* The roots are symmetric, so we only find half of them. */
	xm=0.5*(x2+x1);
	xl=0.5*(x2-x1);
	for (i=1;i<=m;i++) { /* Loop over the desired roots. */
		z=cos(3.141592654*(i-0.25)/(n+0.5));
		/* Starting with the above approximation to the ith root, we enter */
		/* the main loop of refinement by Newton's method.                 */
		do {
			p1=1.0;
			p2=0.0;
			for (j=1;j<=n;j++) { /* Recurrence to get Legendre polynomial. */
				p3=p2;
				p2=p1;
				p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
			}
			/* p1 is now the desired Legendre polynomial. We next compute */
			/* pp, its derivative, by a standard relation involving also  */
			/* p2, the polynomial of one lower order.                     */
			pp=n*(z*p1-p2)/(z*z-1.0);
			z1=z;
			z=z1-p1/pp; /* Newton's method. */
		} while (fabs(z-z1) > EPS);
		x[i-1]=xm-xl*z;      /* Scale the root to the desired interval, */
		x[n-i]=xm+xl*z;  /* and put in its symmetric counterpart.   */
		w[i-1]=2.0*xl/((1.0-z*z)*pp*pp); /* Compute the weight             */
		w[n-i]=w[i-1];                 /* and its symmetric counterpart. */
	}
}


/*******************************************************************************
Integrate a 1D array by the gaussian quadrature given the weights and the 
array of points in the quadrature abcises + dimension of the array
Author: Andres Vicente Arevalo 
*******************************************************************************/
double num_gaus_quad(double y[], double weigths[], int nn){
    double result = 0.;

    for (int i = 0; i < nn; i++){
        result = result + y[i]*weigths[i];
    }

    return result;    
}


/*******************************************************************************
Integrate a 1D array by the composite trapezoidal rule
Author: Andres Vicente Arevalo 
*******************************************************************************/
double trapezoidal(double y[], double x[], int n){
    double Integral = 0.;

    for (int i = 1; i < n; i++){
        Integral = Integral + (y[i-1] + y[i])*(x[i]-x[i-1])/2.;
    }
    return Integral;
}