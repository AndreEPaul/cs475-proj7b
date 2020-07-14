#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#ifndef NUMT
#define NUMT    16
#endif

using std::ifstream;

int main(){
    
    ifstream inputFile;
    inputFile.open("signal.txt");
    
    omp_set_num_threads(NUMT);

    if( !inputFile )
    {
        fprintf( stderr, "Cannot open file 'signal.txt'\n" );
        exit( 1 );
    }
    
    int Size;
    inputFile >> Size;

    float *A =     new float[ 2*Size ];
    float *Sums  = new float[ 1*Size ];
    
    for( int i = 0; i < Size; i++ )
    {
        inputFile >> A[i];
        A[i+Size] = A[i];		// duplicate the array
    }
    
    inputFile.close();

    float sum;

    double time0 = omp_get_wtime();
    for( int shift = 0; shift < Size; shift++ )
    {
        sum = 0.;
        for( int i = 0; i < Size; i++ )
        {
            sum += A[i] * A[i + shift];
        }
        Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }
    double time1 = omp_get_wtime();

    double time2 = omp_get_wtime();
    #pragma omp parallel for default(none) 
    for( int shift = 0; shift < Size; shift++ )
    {
        sum = 0.;
        for( int i = 0; i < Size; i++ )
        {
            sum += A[i] * A[i + shift];
        }
        Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }
    double time3 = omp_get_wtime();

    double megaAddsNothing = (double) Size*Size / (time1-time0) / 1000000.;
    double megaAdds8Threads = (double) Size*Size / (time3-time2) / 1000000.;

    // Provided intrinsic code.
    float SimdMulSum( float *a, float *b, int len ){
        float sum[4] = { 0., 0., 0., 0. };
        int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;
        register float *pa = a;
        register float *pb = b;

        __m128 ss = _mm_loadu_ps( &sum[0] );
        for( int i = 0; i < limit; i += SSE_WIDTH )
        {
            ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
            pa += SSE_WIDTH;
            pb += SSE_WIDTH;
        }
        _mm_storeu_ps( &sum[0], ss );

        for( int i = limit; i < len; i++ )
        {
            sum[0] += a[i] * b[i];
        }

        return sum[0] + sum[1] + sum[2] + sum[3];
    }

    time0 = omp_get_wtime();
    for( int shift = 0; shift < Size; shift++ )
    {
        Sums[shift] = SimdMulSum(&A[shift], &A[shift + Size], Size);	// note the "fix #2" from false sharing if you are using OpenMP
    }
    time1 = omp_get_wtime();

    double megaAddsSIMD = (double) Size*Size / (time1-time0) / 1000000.;

    printf("\n%lf,%lf,%lf\n", megaAddsNothing, megaAdds8Threads, megaAddsSIMD);

    delete [] A;
    delete [] Sums;

    return 0;
}