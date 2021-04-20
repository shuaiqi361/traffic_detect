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
//#include <iostream>
/* ===================================================================== */
/** 
 * Compute the CMM measure of given contours
 * @param point on the first curve ia, ib
 * @param point on the second curve ja, jb, both are double array
 * @param n1 > n2, length of each contour. Pre-Assumption: first curve is the longer one (n1>n2)
 * @return double value, cmm
 */
extern "C" // required when using C++ compiler
double cmm_truncate_sides(double *ia, double *ja, double *ib, double *jb, int n1, int n2)
{

    double *D, *C, big, y, bestcost;
    int i, j, yi, ct, stop, N2, noTpnts;
    //int bestct;
    int norot;
    int *YI, *alltrace;
  
    big = 1e+10;
    bestcost = big;
    N2 = 2 * n2;
    noTpnts = n1 + n2; //max no of trace points
    norot = n2 + 1; //no of possible rotations- n2 rotations of B is B again (0 to n2)
   
  
    // Memory allocations 
    D= (double*) calloc(n1*N2, sizeof(double));
    C= (double*) calloc(n1*N2, sizeof(double));
    YI= (int*) calloc(n1*N2, sizeof(int));
    int start_i = 0; int end_i = 0; int start_j = 0; int end_j = 0;

    //allct= (int*) calloc(norot, sizeof(int)); // length of trace for each rotation
    //mincT= (int*) calloc(n1, sizeof(int));    // min col to calculate for each row
    //maxcT= (int*) calloc(n1, sizeof(int));    // max col to calculate for each row
    alltrace= (int*) calloc( norot*noTpnts*2, sizeof(int)); //traces for all rotations
    // alltrace is divided into norot blocks of noTpnts*2 cells each, each block for one rotation
    // again each block has noTpnts slots for two values (i,j)
    // the xth point coordinates (xr,xc) on trace of rth rotation  is saved as follows:
    // xr is in *(alltrace+ r * noTpnts*2 + x*2 +0) 
    // xc is in *(alltrace+ r * noTpnts*2 + x*2 +1)
    
    
    // Calculation (& duplication) of distances in D: shape (n1 * 2x n2)
    for (i=0; i<n1; i++){
        for (j=0; j<n2; j++){
            *(D+j+i*N2) = pow ( pow(*(ia+i)- *(ib+j),2) + pow(*(ja+i)- *(jb+j),2), 0.5);
            *(D+j+n2+i*N2) = *(D+j+i*N2);
        }
    }
            
    // allct keeps length of trace for each rotation
    //for (j=0; j<norot; j++)
    //    *(allct+j)=0;
        
    // finding the best path for first rotation
    //rot=0;
    // DP table is stored in C, stores the cost
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
    int first_leave_last_point = 0;  // if the last point of both contour has been examined
    int temp_status = 0;  // flag to keep track of the current step
    int count_pruned_path = 1;
    while (!stop)
    {
        ct=ct+1;
        temp_status = *(YI+j+i*N2);
        switch (*(YI+j+i*N2))
        {               
            case 1: j=j-1; break;
            case 2: i=i-1; break;
            case 3: i=i-1; j=j-1; break; 
            default: stop=1;
        }
        if ((first_leave_last_point == 0) && (j < n2-1) && (i < n1-1)){
            first_leave_last_point = 1;
            switch (temp_status){
                case 1: end_i = i; end_j = j + 1; break;
                case 2: end_i = i + 1; end_j = j; break;
                case 3: end_i = i + 1; end_j = j + 1; break;
                default: stop=1;
            }
        }
        if (first_leave_last_point == 1) count_pruned_path += 1;

        *(alltrace+ct*2+0)=i; *(alltrace+ct*2+1)=j;
        //if ((i==0) && (j==0))
        //    stop=1;
        if ((i==0) || (j==0)){
            stop=1;
            if ((i == 0) && (j != 0)){
                start_i = i;
                start_j = j - 1;
            }
            if ((i != 0) && (j == 0)){
                start_i = i - 1;
                start_j = j;
            }
            if ((i == 0) && (j == 0)){
                start_i = i;
                start_j = j;
            }

        }

    }
    //*(allct+rot)=ct+1;                    // no of points in best trace for this rotation
    //bestcost= *(C+(n2-1) + (n1-1)*N2);    // cost of best trace so far
    //bestct=ct+1;                          // no of points in best trace so far
    if (start_j + start_i == 0){
        bestcost= *(C+(end_j) + (end_i)*N2);
    }else{
        bestcost= *(C+(end_j) + (end_i)*N2) - *(C+(start_j) + (start_i)*N2);
    }

    //bestct = end_i + end_j - start_i - start_j;
    //std::cout << start_i << ", " << start_j << ", " << end_i << ", " << end_j << std::endl;
    //std::cout << *(C+(end_j) + (end_i)*N2) << ", " << *(C+(start_j) + (start_i)*N2) << ", " << count_pruned_path << std::endl;

    free(D); free(C);  free(YI); 
    //free(allct); free(mincT); free(maxcT);
    free(alltrace);

    double bestcostp = bestcost / count_pruned_path;
    
    return bestcostp;
}

