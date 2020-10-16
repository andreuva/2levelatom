# -*- coding: utf-8 -*-

######################################################################
######################################################################
######################################################################
#                                                                    #
# jsymbols.py                                                        #
#                                                                    #
# Tanaus\'u del Pino Alem\'an                                        #
#   Instituto de Astrof\'isica de Canarias                           #
#                                                                    #
######################################################################
######################################################################
#                                                                    #
# Class to compute 3, 6, and 9 j-symbols                             #
#                                                                    #
######################################################################
######################################################################
#                                                                    #
######################################################################
######################################################################
#                                                                    #
#  26/10/2018 - V1.0.0 - First version. (TdPA)                       #
#                                                                    #
######################################################################
######################################################################
######################################################################

import math

######################################################################
######################################################################
######################################################################

class jsymbols():
    ''' Class to compute 3, 6, and 9 j-symbols
    '''

######################################################################
######################################################################
######################################################################

    def __init__(self, initialize=None):
        ''' Class initializer
        '''

        # Initialize factorial and sign list
        self.__fact = [0.]
        self.__sign = [1.]

        # Initialize factorial list up to the specified value
        if initialize is not None:
            if isinstance(initialize, int) or \
               isinstance(initialize, float):
                a = self.logfct(initialize)
                a = self.sign(initialize)

######################################################################
######################################################################
######################################################################

    def logfct(self, val):
        ''' Returns the logarithm of the factorial of val
        '''

        # Real val
        ival = int(math.fabs(val))

        # Compute and store up to val
        while len(self.__fact) <= ival:

            self.__fact.append(self.__fact[-1] + \
                               math.log(float(len(self.__fact))))

        return self.__fact[ival]

######################################################################
######################################################################
######################################################################

    def sign(self, val):
        ''' Returns the sign of val
        '''

        # Real val
        ival = int(math.fabs(val))

        # Compute and store up to val
        while len(self.__sign) <= ival:

            self.__sign.append(self.__sign[-1] * -1.)

        return self.__sign[ival]

######################################################################
######################################################################
######################################################################

    def __fn1(self, j1, j2, j3, m1, m2, m3):
        ''' Auxiliar used in 3j calculations
        '''

        l1 = int(round(j1 + j2 - j3))
        l2 = int(round(j2 + j3 - j1))
        l3 = int(round(j3 + j1 - j2))
        l4 = int(round(j1 + j2 + j3) + 1)
        l5 = int(round(j1 + m1))
        l6 = int(round(j1 - m1))
        l7 = int(round(j2 + m2))
        l8 = int(round(j2 - m2))
        l9 = int(round(j3 + m3))
        l10 = int(round(j3 - m3))

        fn1 = 0.5*(self.logfct(l1) + self.logfct(l2) + \
                   self.logfct(l3) - self.logfct(l4) + \
                   self.logfct(l5) + self.logfct(l6) + \
                   self.logfct(l7) + self.logfct(l8) + \
                   self.logfct(l9) + self.logfct(l10))
                
        return fn1

######################################################################
######################################################################
######################################################################

    def __fn2(self, ij1, ij2, ij3):
        ''' Auxiliar used in 3j calculations
        '''

        l1 = int(round(ij1+ij2-ij3))/2
        l2 = int(round(ij2+ij3-ij1))/2
        l3 = int(round(ij3+ij1-ij2))/2
        l4 = int(round(ij1+ij2+ij3))/2 + 1

        fn2 = 0.5*(self.logfct(l1) + self.logfct(l2) + \
                   self.logfct(l3) - self.logfct(l4))

        return fn2

######################################################################
######################################################################
######################################################################

    def j3(self, j1, j2, j3, m1, m2, m3):
        ''' Compute 3j symbol
        '''

        # Initialize value
        js3 = 0.0

        # Conver to integer combinations
        ij1 = int(round(j1 + j1))
        ij2 = int(round(j2 + j2))
        ij3 = int(round(j3 + j3))

        # Selection rules
        if ij1 + ij2 - ij3 < 0:
            return js3
        if ij2 + ij3 - ij1 < 0:
            return js3
        if ij3 + ij1 - ij2 < 0:
            return js3

        # Conver to integer combinations
        im1 = int(round(m1 + m1))
        im2 = int(round(m2 + m2))
        im3 = int(round(m3 + m3))

        # Selection rules
        if im1 + im2 + im3 != 0:
            return js3
        if math.fabs(im1) > ij1:
            return js3
        if math.fabs(im2) > ij2:
            return js3
        if math.fabs(im3) > ij3:
            return js3

        # Get minimum index to run from
        kmin = (ij3 - ij1 - im2)/2
        kmin1 = kmin
        kmin2 = (ij3 - ij2 + im1)/2
        kmin = max(-1*min(kmin,kmin2),0)

        # Get maximum index to run to
        kmax = int(round(j1 + j2 - j3))
        kmax1 = kmax
        kmax2 = int(round(j1 - m1))
        kmax3 = int(round(j2 + m2))
        kmax = min([kmax,kmax2,kmax3])

        if kmin <= kmax:

          term1 = self.__fn1(j1,j2,j3,m1,m2,m3)

          sgn = self.sign((ij1 - ij2 - im3)/2)

          for i in range(kmin,kmax+1):

              term2 = self.logfct(i) + self.logfct(kmin1+i) + \
                      self.logfct(kmin2+i) + self.logfct(kmax1-i) + \
                      self.logfct(kmax2-i) + self.logfct(kmax3-i)
              js3 = self.sign(i)*math.exp(term1-term2) + js3

          js3 = sgn*js3

        return js3

######################################################################
######################################################################
######################################################################

    def j6(self, j11, j12, j13, j21, j22, j23):
        ''' Compute 6j symbol
        '''

        # Initialize value
        js6 = 0.0

        # Conver to integer combinations
        ij1 = int(round(j11 + j11))
        ij2 = int(round(j12 + j12))
        ij3 = int(round(j13 + j13))
        ij4 = int(round(j21 + j21))
        ij5 = int(round(j22 + j22))
        ij6 = int(round(j23 + j23))

        ijm1 = (ij1 + ij2 + ij3)/2
        ijm2 = (ij1 + ij5 + ij6)/2
        ijm3 = (ij4 + ij2 + ij6)/2
        ijm4 = (ij4 + ij5 + ij3)/2

        ijm = ijm1

        ijm = max([ijm,ijm2,ijm3,ijm4])

        ijx1 = (ij1 + ij2 + ij4 + ij5)/2
        ijx2 = (ij2 + ij3 + ij5 + ij6)/2
        ijx3 = (ij3 + ij1 + ij6 + ij4)/2

        ijx = ijx1

        ijx = min([ijx,ijx2,ijx3])

        if ijm <= ijx:

          term1 = self.__fn2(ij1,ij2,ij3) + \
                  self.__fn2(ij1,ij5,ij6) + \
                  self.__fn2(ij4,ij2,ij6) + \
                  self.__fn2(ij4,ij5,ij3)

          for i in range(ijm,ijx+1):

              term2 = self.logfct(i+1) - self.logfct(i-ijm1) - \
                      self.logfct(i-ijm2) - self.logfct(i-ijm3) - \
                      self.logfct(i-ijm4) - self.logfct(ijx1-i) - \
                      self.logfct(ijx2-i) - self.logfct(ijx3-i)
              js6 = self.sign(i)*math.exp(term1+term2) + js6

        return js6

######################################################################
######################################################################
######################################################################

    def j9(self, j11, j12, j13, j21, j22, j23, j31, j32, j33):
        ''' Compute 9j symbol
        '''

        # Initialize value
        js9 = 0.0

        # Conver to integer combinations
        ij11 = int(round(j11 + j11))
        ij12 = int(round(j12 + j12))
        ij13 = int(round(j13 + j13))
        ij21 = int(round(j21 + j21))
        ij22 = int(round(j22 + j22))
        ij23 = int(round(j23 + j23))
        ij31 = int(round(j31 + j31))
        ij32 = int(round(j32 + j32))
        ij33 = int(round(j33 + j33))

        kmin1 = int(round(math.fabs(ij11 - ij33)))
        kmin2 = int(round(math.fabs(ij32 - ij21)))
        kmin3 = int(round(math.fabs(ij23 - ij12)))

        kmin1 = max([kmin1,kmin2,kmin3])

        kmax1 = ij11 + ij33
        kmax2 = ij32 + ij21
        kmax3 = ij23 + ij12

        kmax1 = min(kmax1,kmax2,kmax3)

        if kmin1 <= kmax1:

            for k in range(kmin1,kmax1+1,2):

                hk = 0.5*float(k)

                js9 = self.sign(k)*float(k+1)* \
                      self.j6(j11,j21,j31,j32,j33,hk)* \
                      self.j6(j12,j22,j32,j21,hk,j23)* \
                      self.j6(j13,j23,j33,hk,j11,j12) + js9

        return js9

######################################################################
######################################################################
######################################################################
