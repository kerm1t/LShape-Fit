#ifndef _L_SHAPED_FIT_X86_H
#define _L_SHAPED_FIT_X86_H

#include <immintrin.h>
#include <cpuid.h>
#include "lshaped_fit.h"

bool cpuHasAVX2() {
    int info[4];
    __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
    return (info[1] & (1 << 5)) != 0;    // AVX2 bit
}

void projectPointsAVX2(const geo::Mat& M, geo::Mat& c1, geo::Mat& c2,
                       double c, double s)
{
    const int n = M.rows;
    __m256d vc = _mm256_set1_pd(c);
    __m256d vs = _mm256_set1_pd(s);
    __m256d vns = _mm256_set1_pd(-s);

    int i = 0;
    for (; i <= n - 4; i += 4) {
        // Load x values
        __m256d x = _mm256_set_pd(
            M.at(i+3,0), M.at(i+2,0), M.at(i+1,0), M.at(i,0)
        );

        // Load y values
        __m256d y = _mm256_set_pd(
            M.at(i+3,1), M.at(i+2,1), M.at(i+1,1), M.at(i,1)
        );

        // c1 = x*c + y*s
        __m256d c1v = _mm256_add_pd(_mm256_mul_pd(x, vc),
                                    _mm256_mul_pd(y, vs));

        // c2 = -x*s + y*c
        __m256d c2v = _mm256_add_pd(_mm256_mul_pd(x, vns),
                                    _mm256_mul_pd(y, vc));

        // store results
        double out1[4], out2[4];
        _mm256_storeu_pd(out1, c1v);
        _mm256_storeu_pd(out2, c2v);

        for (int k = 0; k < 4; k++) {
            c1.at(i + k, 0) = out1[k];
            c2.at(i + k, 0) = out2[k];
        }
    }

    // tail
    for (; i < n; i++) {
        double x = M.at(i,0);
        double y = M.at(i,1);
        c1.at(i,0) = x * c + y * s;
        c2.at(i,0) = -x * s + y * c;
    }
}
#endif // _L_SHAPED_FIT_X86_H
