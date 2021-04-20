/*=================================================================
 * Vida Movahedi elderlab.yorku.ca
 * cmm.cpp
 * takes the coordinates of samples on two curves
 *
 * and returns the contour mapping distance normalized by size of trace
 *
 * Using Maes' paper for faster Dynamic programming
 *
 * This is a MEX-file for MATLAB.
 *=============================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
/* ===================================================================== */
/** 
 * Compute the CMM measure of given contours
 * @param point on the first curve ia, ib
 * @param point on the second curve ja, jb, both are double array
 * @param n1 > n2, length of each contour. Pre-Assumption: first curve is the longer one (n1>n2)
 * @return double value, cmm
 */
extern "C" // required when using C++ compiler
double cmm(double *ia, double *ja, double *ib, double *jb, int n1, int n2)
{

    double *D, *C, big, y, bestcost ;
    int i,j, rot, yi, ct, stop, jj, kk, N2, noTpnts, norot;
    int rotlow, rothig, preL, preH,ti, minc, maxc,qu, jpre, bestct;
    int *YI, *alltrace, *allct, *mincT, *maxcT;
  
    big= 1e+10;
    bestcost=big;
    N2= 2*n2; 
    noTpnts=n1+n2; //max no of trace points
    norot= n2+1; //no of possible rotations- n2 rotations of B is B again (0 to n2)
   
  
    // Memory allocations 
    D= (double*) calloc(n1*N2, sizeof(double)); 
    C= (double*) calloc(n1*N2, sizeof(double));
    YI= (int*) calloc(n1*N2, sizeof(int)); 

    allct= (int*) calloc(norot, sizeof(int)); // length of trace for each rotation
    mincT= (int*) calloc(n1, sizeof(int));    // min col to calculate for each row
    maxcT= (int*) calloc(n1, sizeof(int));    // max col to calculate for each row
    alltrace= (int*) calloc( norot*noTpnts*2, sizeof(int)); //traces for all rotations
    // alltrace is divided into norot blocks of noTpnts*2 cells each, each block for one rotation
    // again each block has noTpnts slots for two values (i,j)
    // the xth point coordinates (xr,xc) on trace of rth rotation  is saved as follows:
    // xr is in *(alltrace+ r * noTpnts*2 + x*2 +0) 
    // xc is in *(alltrace+ r * noTpnts*2 + x*2 +1)
    
    
    // Calculation (& duplication) of distances in D
    for (i=0; i<n1; i++){
        for (j=0; j<n2; j++){
            *(D+j+i*N2) = pow ( pow(*(ia+i)- *(ib+j),2) + pow(*(ja+i)- *(jb+j),2), 0.5);
            *(D+j+n2+i*N2) = *(D+j+i*N2);
        }
    }
            
    // allct keeps length of trace for each rotation
    for (j=0; j<norot; j++)
        *(allct+j)=0;
        
    // finding the best path for first rotation
    rot=0;
    // DP table is stored in C,
    // Best move is stored in YI (1: horiz-move, 2:vertical-move, 3: diag-move)
    *(C+0)=*(D+0); 
    for (j=1; j<n2; j++){
        *(C+j)= *(C+j-1) + *(D+j); *(YI+j)=1; } // first row
    for (i=1; i<n1; i++){
        *(C+i*N2)= *(C+(i-1)*N2) + *(D+i*N2); *(YI+i*N2)=2; } // first col
    
    for (i=1; i<n1; i++) { // other rows & cols    
        for (j=1; j<n2; j++) {
            y=*(C+(j-1)+i*N2); yi=1;
            if (*(C+j+(i-1)*N2)<y) {y= *(C+j+(i-1)*N2); yi=2; }
            if (*(C+(j-1)+(i-1)*N2)<y) {y=*(C+(j-1)+(i-1)*N2); yi=3; }
            *(C+j+i*N2)= *(D+j+i*N2) + y;
            *(YI+j+i*N2)= yi;
        }
    }
    
    //backtrace to get path
    j=n2-1; ct=0; i=n1-1;
    *(alltrace+0)=i; *(alltrace+1)=j;stop=0; 
    while (!stop)
    {
        ct=ct+1;
        switch (*(YI+j+i*N2))
        {               
            case 1: j=j-1; break;
            case 2: i=i-1; break;
            case 3: i=i-1; j=j-1; break; 
            default: stop=1;
        }
        *(alltrace+ct*2+0)=i; *(alltrace+ct*2+1)=j; 
        if ((i==0) && (j==0)) 
            stop=1;
    }
    *(allct+rot)=ct+1;                    // no of points in best trace for this rotation
    bestcost= *(C+(n2-1) + (n1-1)*N2);    // cost of best trace so far
    bestct=ct+1;                          // no of points in best trace so far
        
    //printf(" First rotation executed with bestcost=%.2f\n", bestcost);

    // Replicate the path for rot= n2
    rot=n2;
    for (i=0; i<=ct; i++)
    {
        *(alltrace +rot*noTpnts*2 +i*2 +0) = *(alltrace +i*2 +0);
        *(alltrace +rot*noTpnts*2 +i*2 +1) = *(alltrace +i*2 +1) + n2;
    }
    *(allct+rot)=ct+1;
    
    
    // Calculating for the rest of rotations possible- using Maes algorithm
    qu= (int)ceil(log(n2) /log(2));
    for (jj=1; jj<=qu; jj++){        
        for (kk=1; kk<= pow(2,(jj-1)); kk++) {            
            rot=(int)floor((2 * kk-1)* n2/pow(2,jj));
            rotlow=(int)floor( (2 * kk -2)* n2/pow(2,jj));
            rothig=(int)floor( 2 * kk * n2/pow(2,jj));
            if ((rot!=rotlow) && (rot !=rothig) && (*(allct+rot)==0))
            {
               //printf("rot=%d\n", rot);
               
               //mincT : the lowest col in each row that trace can go on
               //maxcT : the highest col in each row that trace can go on
               //setting mincT and maxcT:                         
               *(mincT+0)=rotlow; 
               *(maxcT+0)=rothig; 
               for (ti=1; ti<n1; ti++){ //each row
                   *(mincT+ti)=100000; *(maxcT+ti)=0;
               }
               for (ti=0;ti<*(allct+rotlow); ti++){ // looking at entries in trace of rotlow
                   if (*(alltrace+ rotlow * noTpnts *2 + ti*2 +1)<= *(mincT+ *(alltrace + rotlow * noTpnts *2 + ti*2 +0))){
                      *(mincT+ *(alltrace + rotlow*noTpnts*2 +ti*2 +0))= *(alltrace +rotlow * noTpnts*2 +ti*2 +1);
                   }
               }
               for (ti=0; ti<*(allct+rothig); ti++){// looking at entries in trace of rothig
                   if (*(alltrace+ rothig*noTpnts*2 +ti*2 +1)>=*(maxcT+ *(alltrace + rothig* noTpnts*2 +ti*2+0))){ 
                      *(maxcT+ *(alltrace+rothig*noTpnts*2+ti*2+0))= *(alltrace+rothig*noTpnts*2+ti*2+1);
                   }
               }
               
               // Now calculation of entries in table
               // first point- matching point 0 on A to point rot on B
               *(C+rot) = *(D+rot); 
               // first row (i=0)
               for (j=rot+1; j<=*(maxcT+0); j++) { 
                   *(C+j)= *(C+j-1) + *(D+j); *(YI+j)=1; 
               }               
               
               // the next rows
               preL=rot; preH=*(maxcT+0); //range of numbers changed in previous row
               for (i=1;i<n1;i++) {
                   minc= *(mincT+i); if (minc<preL) minc=preL;
                   maxc= *(maxcT+i); jpre=maxc;
                   for (j=minc; j<=maxc; j++){
                       yi=4; y=big;
                       if (j>minc) {y= *(C+i*N2+j-1); yi=1; }
                       if ((j>= preL) && (j<=preH)) {
                       if (*(C+(i-1)*N2+j)<y) { y= *(C+(i-1)*N2+j); yi=2; }}
                       if (((j-1)>=preL) && ((j-1)<=preH)) {
                          if (*(C+(i-1)*N2+(j-1))<y) { y= *(C+(i-1)*N2+ j-1); yi=3; }}
                       if (yi!=4) {
                          if (j<jpre) jpre=j; // to mark the first column changed
                          *(C+i*N2+j)= *(D+i*N2+j) + y;
                          *(YI+i*N2+j)=yi;
                       }                       
                   }
                   preL= jpre; //minc;
                   preH= maxc;
               }
               
               // backtrace to save trace        
               j=n2+rot-1;ct=0; i=n1-1;
               *(alltrace+rot*noTpnts*2+ct*2+0)=i; *(alltrace+rot*noTpnts*2+ct*2+1)=j; 
               stop=0;
               while (!stop)
               {
                     ct=ct+1;
                     switch (*(YI+j+i*N2)){  
                            case 1: j=j-1; break;
                            case 2: i=i-1; break;
                            case 3: i=i-1; j=j-1; break; 
                            default: stop=1;
                     }
                     *(alltrace+rot*noTpnts*2+ct*2+0)=i; 
                     *(alltrace+rot*noTpnts*2+ct*2+1)=j; 
                     if ((i==0) && (j==rot)) stop=1;
               }
               *(allct+rot)=ct+1;
              
               // save bestcost so far
               if (*(C+(n1-1)*N2 +(n2+rot-1)) < bestcost) {
                  bestcost=*(C+(n1-1)*N2 +(n2+rot-1)); 
                  bestct=ct+1;
                  }
            }
        }
    }

    
    free(D); free(C);  free(YI); 
    free(allct); free(mincT); free(maxcT); free(alltrace);

    double bestcostp = bestcost / bestct;
    
    return bestcostp;
}

