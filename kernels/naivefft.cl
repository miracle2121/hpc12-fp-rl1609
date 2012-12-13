
/*
  Naive fft implementation. 
 */

#define TWOPI 6.28318530718

__kernel void naivefft( __global const float2* src, 
	                    __global float2* dst, 
	                    const unsigned int n) {
    const float ph = -TWOPI / n;
    const int gid = get_global_id(0);

    float2 res = (float2) (0.0f, 0.0f);
    for (int k = 0; k < n; k++) {
        const float2 t = src[k];

        const float val = ph * k * gid;
        float cs;
	    float sn = sincos(val, &cs);
        res.x += t.x * cs - t.y * sn;
        res.y += t.y * cs + t.x * sn;
    }

    dst[gid] = res;
}

