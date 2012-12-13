/* 
   Stockham's implementation of radix-8 fft.
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
    sn = sincos(alpha, &cs);
    return (float2)(cs, sn);
}

float2 twiddle(float2 a) { 
    return (float2)(a.y, -a.x); 
}

float2 square(float2 a) { 
    return (float2)(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y); 
}

__constant float SQRT_2 = 0.707106781188f;

float2 mul_p1q4(float2 a) { 
    return (float2)(SQRT_2) * (float2)(a.x + a.y, -a.x + a.y); 
}


float2 mul_p3q4(float2 a) { 
    return (float2)(SQRT_2) * (float2)(-a.x + a.y, -a.x - a.y); 
}

__kernel void fft_radix8(__global float2* src, 
                         __global float2* dst,
                         const int p,
                         const int t) {

    const int gid = get_global_id(0);
    const int k = gid & (p - 1);
    src += gid;
    dst += ((gid - k) << 3) + k; 

    const float theta = -TWOPI * k / (4 * p);

    float2 tw = sincos_float2(theta); // W
    float2 a0 = src[0];
    float2 a1 = mul_float2(tw, src[t]);
    float2 a2 = src[2 * t];
    float2 a3 = mul_float2(tw, src[3 * t]);
    float2 a4 = src[4 * t];
    float2 a5 = mul_float2(tw, src[5 * t]);
    float2 a6 = src[6 * t];
    float2 a7 = mul_float2(tw, src[7 * t]);

    tw = square(tw); // W^2
    a2 = mul_float2(tw, a2);
    a3 = mul_float2(tw, a3);
    a6 = mul_float2(tw, a6);
    a7 = mul_float2(tw, a7);
    tw = square(tw); // W^4
    a4 = mul_float2(tw, a4);
    a5 = mul_float2(tw, a5);
    a6 = mul_float2(tw, a6);
    a7 = mul_float2(tw, a7);

    float2 b0 = a0 + a4;
    float2 b4 = a0 - a4;
    float2 b1 = a1 + a5; 
    float2 b5 = mul_p1q4(a1 - a5);
    float2 b2 = a2 + a6;
    float2 b6 = twiddle(a2 - a6);
    float2 b3 = a3 + a7;
    float2 b7 = mul_p3q4(a3 - a7);

    a0 = b0 + b2;
    a2 = b0 - b2;
    a1 = b1 + b3;
    a3 = twiddle(b1 - b3);
    a4 = b4 + b6;
    a6 = b4 - b6;
    a5 = b5 + b7;
    a7 = twiddle(b5 - b7);

    dst[0] = a0 + a1;
    dst[p] = a4 + a5;
    dst[2 * p] = a2 + a3;
    dst[3 * p] = a6 + a7;
    dst[4 * p] = a0 - a1;
    dst[5 * p] = a4 - a5;
    dst[6 * p] = a2 - a3;
    dst[7 * p] = a6 - a7;
}
