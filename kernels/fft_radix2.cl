/* 
   Stockham's implementation of radix-2 fft.
 */

#define TWOPI 6.28318530718

__kernel void fft_radix2(__global float2* src, /*input array*/ 
                         __global float2* dst, /*output array*/
                         const int p,          /*block size*/
                         const int t) {        /*number of threads*/

    const int gid = get_global_id(0);
    const int k = gid & (p - 1);
    src += gid;
    dst += (gid << 1) - k; 

    const float2 in1 = src[0];
    const float2 in2 = src[t];
    
    const float theta = -TWOPI * k / p;
    float cs;
    float sn = sincos(theta, &cs);
    const float2 temp = (float2) (in2.x * cs - in2.y * sn,
                                  in2.y * cs + in2.x * sn);

    dst[0] = in1 + temp;
    dst[p] = in1 - temp;
}
