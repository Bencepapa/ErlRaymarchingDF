# ErlRaymarchingDF
Erlang port of Inigo Quilez's Raymarching distance fields reference implementation from Shadertoy

# Usage
Example usage (creates a 160x120 .ppm):
```erlang
src $ erl
Erlang/OTP 20 [erts-9.2] [source] [64-bit] [smp:4:4] [ds:4:4:10] [async-threads:10] [kernel-poll:false]

Eshell V9.2  (abort with ^G)
1> c(raymarch).
raymarch.erl:170: Warning: function write_pixels_to_ppm/5 is unused
raymarch.erl:229: Warning: function yz/1 is unused
raymarch.erl:269: Warning: function vec2/1 is unused
raymarch.erl:388: Warning: function opU/2 is unused
raymarch.erl:447: Warning: function sdEquilateralTriangle/1 is unused
{ok,raymarch}
2> raymarch:test().
all workers have been spawned
master is done
preparing for binary ppm
file "bmini.ppm" opened
ok
3> 
```

# Output
![rendered 2560x1440 image](/renderedpics/b2560x1440.png)

# Links
* http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
* https://www.shadertoy.com/view/Xds3zN


