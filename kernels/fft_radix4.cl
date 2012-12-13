/* 
   Stockham's implementation of radix-4 fft.
 */

#define TWOPI 6.28318530718

float2 mul_float2(float2 a, float2 b){ 
    float2 res; 
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res; 
}

float2 sincos_float2(float alpha) {
    float cs, sn;
    sn = sincos(alpha, &cs);  // sincos
    return (float2)(cs, sn);
}

float2 twiddle(float2 a) { 
    return (float2)(a.y, -a.x); 
}

float2 square(float2 a) { 
    return (float2)(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y); 
}

__kernel void fft_radix4(__global float2* src, 
                         __global float2* dst,
                         const int p,
                         const int t) {

    const int gid = get_global_id(0);
    const int k = gid & (p - 1);
    src += gid;
    dst += ((gid - k) << 2) + k; 

    const float theta = -TWOPI * k / (2 * p);
    float2 tw = sincos_float2(theta);
    float2 a0 = src[0];
    float2 a1 = mul_float2(tw, src[t]);
    float2 a2 = src[2 * t];
    float2 a3 = mul_float2(tw, src[3 * t]);
    tw = square(tw);
    a2 = mul_float2(tw, a2);
    a3 = mul_float2(tw, a3);

    float2 b0 = a0 + a2;
    float2 b1 = a0 - a2;
    float2 b2 = a1 + a3;
    float2 b3 = twiddle(a1 - a3);

    dst[0] = b0 + b2;
    dst[p] = b1 + b3;
    dst[2 * p] = b0 - b2;
    dst[3 * p] = b1 - b3; 
}
