-module(raymarch).
-export([t/0, test/0, testbig/0, testbigAA/0, testHD/0, go/4, master/2, worker/5]).


t() ->
    go(8,6, 0, "bmicro.ppm").

test() ->
    go(160,120, 0, "bmini.ppm").

testbig() ->
    go(640,480, 0, "bbig.ppm").

testbigAA() ->
    go(640,480, 4, "bbigAA.ppm").

testHD() ->
    go(1980,1080, 4, "bhd.ppm").

go(Width, Height, AA, Filename) ->
    %write_pixels_to_ppm(
    write_pixels_to_ppm_binary(
      Width,
      Height,
      255,
        raymarch(Width, Height, AA),
      Filename).

% Spawn master and workers
raymarch(Width, Height, AA)
  when Width > 0, Height > 0 ->
    AspectRatio = Height/Width,
    Master_PID = spawn(?MODULE, master, [self(), Width*Height]),
    lists:flatmap(
      fun(Y) ->
	      lists:map(
		fun(X) ->
			% coordinates passed as a percentage
			spawn(?MODULE, worker,
			  [Master_PID, X+Y*Width, {X/Width, Y/Height},
%                {AspectRatio, {-0.25, 1.3, -3.6}},
                {AspectRatio, {+2.6, 1.3, 3.7}},
              AA]
            )
        end,
		lists:seq(0, Width - 1)) end,
      lists:seq(0, Height - 1)),
    io:format("all workers have been spawned~n", []),
    receive
	Final_pixel_list ->
	    Final_pixel_list
    end.

master(Program_PID, Pixel_count) ->
    master(Program_PID, Pixel_count, []).
master(Program_PID, 0, Pixel_list) ->
    io:format("master is done~n", []),
    Program_PID ! lists:keysort(1, Pixel_list);
% assumes all workers eventually return a good value
master(Program_PID, Pixel_count, Pixel_list) ->
    receive
	Pixel_tuple ->
	    master(Program_PID, Pixel_count-1, [Pixel_tuple|Pixel_list])
    end.


% assumes X and Y are percentages of the screen dimensions
worker(Master_PID, Pixel_num, {X, Y}, {AspectRatio, Camera}, _AA) ->
    Master_PID ! {Pixel_num, %{X,Y,1.0-Y}}.
          render(
              Camera,
              %normalize({math:cos(X-0.5 + math:pi()/2), -math:sin(Y-0.5)*AspectRatio-math:pi()/10, 0.5})
              normalize({math:cos(X-0.5 - math:pi()/2*1.50), -math:sin((Y-0.5)*AspectRatio + math:pi()/8), math:sin(X-0.5 - math:pi()/2*1.50)})
          )}.

render( Or, Dir ) -> %Origin and Direction
    Col = vecadd({0.7, 0.9, 1.0}, y(Dir)*0.8),
    {T, M} = castRay(Or, Dir),
    Res = if
        M < -0.5 ->
            Col; %background
        true ->
            Pos = vecadd( Or, vecmult(Dir, T) ),
            Nor = calcNormal( Pos ),
            Ref = reflect( Dir, Nor ),

            %material
            Col2 = if 
                M > 1.5 ->
                    %vecmult({0.05,0.08,0.10}, M-1.0);
                    vecfunc(vecmult({0.05,0.08,0.10}, M-1.0),
                        fun(A) -> 0.45 + 0.35 * math:sin(A) end);
                true ->
                    F = checkersBox( vecmult(xz(Pos), 5.0) ),
                    vec3(0.3 + F * 0.1)
            end,
            
            % lighting
            Occ = calcAO( Pos, Nor ),
            Lig = normalize( {-0.4, 0.7, -0.6} ),
            Hal = normalize( vecsub( Lig, Dir ) ),
            Amb = clamp( 0.5+0.5*y(Nor), 0.0, 1.0 ),
            Dif = clamp( dot( Nor, Lig ), 0.0, 1.0 ),
            Bac = clamp( dot( Nor, normalize({-x(Lig),0.0,-z(Lig)})), 0.0, 1.0 )*clamp( 1.0-y(Pos),0.0,1.0),
            Dom = smoothstep( -0.1, 0.1, y(Ref) ),
            Fre = pow( clamp(1.0+dot(Nor,Dir),0.0,1.0), 2.0 ),

            %TODO
            DifShad = Dif * calcSoftshadow( Pos, Lig, 0.02, 2.5 ),
            DomShad = Dom * calcSoftshadow( Pos, Ref, 0.02, 2.5 ),

            Spe = math:pow( clamp( dot( Nor, Hal ), 0.0, 1.0 ), 16.0)*
                    DifShad *
                    (0.04 + 0.96*math:pow( clamp(1.0+dot(Hal,Dir),0.0,1.0), 5.0 )),

            Lin = vecadd(
                [
                    vecmult({1.00, 0.80, 0.55}, 1.3*DifShad),
                    vecmult({0.40, 0.60, 1.00}, Occ*Amb*1.40),
                    vecmult({0.40, 0.60, 1.00}, Occ*DomShad*0.50),
                    vecmult({0.25, 0.25, 0.25}, Occ*Bac*0.50),
                    vecmult({1.00, 1.00, 1.00}, Occ*Fre*0.25)
                ]),
            Col3 = vecadd( vecmult(Col2,Lin), vecmult( {1.00, 0.90, 0.70}, 10.00*Spe) ),
            mix( Col3, {0.8,0.9,1.0}, 1.0-math:exp( -0.0002*math:pow(T,3) ) )
            %normalize(Col2)
        end,

    Res2 = clamp(Res, 0.0, 1.0),
	% gamma
    vecfunc(Res2, fun(A) -> math:pow(A, 0.4545) end).

castRay( _Or, _Dir, T, _Max, M, 0) ->
    {T, M};
castRay( _Or, _Dir, T, Max, _M, _Depth) when T>Max ->
    {T, -1.0};
castRay( Or, Dir, T, Max, M, Depth) ->
    {Dist, Material} = map( vecadd( Or, vecmult(Dir, T) ) ),
    case Dist =< 0.0004 of
        true -> {T, M};
        _ -> castRay( Or, Dir, T+Dist, Max, Material, Depth-1)
    end. 
castRay( Or, Dir) ->
    castRay( Or, Dir, 1.0, 20, 1.0, 64).



calcNormal( Pos ) ->
    E = {0.5773*0.0005,-0.5773*0.0005},
    normalize( vecadd( [
                vecmult(xyy(E), x(map( vecadd(Pos, xyy(E) )))),
                vecmult(yyx(E), x(map( vecadd(Pos, yyx(E) )))),
                vecmult(yxy(E), x(map( vecadd(Pos, yxy(E) )))),
                vecmult(xxx(E), x(map( vecadd(Pos, xxx(E) ))))
               ])
             ).

calcAO( _Pos, _Nor, 0, Occ, _Sca ) -> Occ;
calcAO( Pos, Nor, I, Occ, Sca ) ->
    HR = 0.01 + 0.12*I/4.0,
    AOPos = vecadd(vecmult(Nor,HR), Pos),
    {DD, _} = map( AOPos),
    calcAO( Pos, Nor, I-1, Occ - (DD-HR)*Sca, Sca*0.95 ).

calcAO( Pos, Nor ) ->
    clamp( 1.0 - 3.0 * calcAO( Pos, Nor, 5, 0.0, 1.0),
            0.0, 1.0).

% assumes Pixels are ordered in a row by row fasion
write_pixels_to_ppm(Width, Height, MaxValue, Pixels, Filename) ->
    case file:open(Filename, write) of
	{ok, IoDevice} ->
	    io:format("file opened~n", []),
	    io:format(IoDevice, "P3~n", []),
	    io:format(IoDevice, "~p ~p~n", [Width, Height]),
	    io:format(IoDevice, "~p~n", [MaxValue]),
        io:format("~p~n", [Pixels]),
	    lists:foreach(
	      fun({_Num, {R, G, B}}) ->
		      io:format(IoDevice, "~p ~p ~p ",
				[lists:min([trunc(R*MaxValue), MaxValue]),
				 lists:min([trunc(G*MaxValue), MaxValue]),
				 lists:min([trunc(B*MaxValue), MaxValue])]) end,
	      Pixels),
	    file:close(IoDevice);
	error ->
	    io:format("error opening file~n", [])
    end.

write_pixels_to_ppm_binary(Width, Height, MaxValue, Pixels, Filename) ->
	io:format("preparing for binary ppm~n", []),
    P = list_to_binary([
        "P6\n", 
        integer_to_list(Width), " ",
        integer_to_list(Height), "\n",
        integer_to_list(MaxValue), "\n",
        lists:map(fun({_, {R,G,B}}) -> [trunc(R*MaxValue),trunc(G*MaxValue),trunc(B*MaxValue)] end, Pixels)
    ]),

    case file:open(Filename, write) of
	{ok, IoDevice} ->
	    io:format("file ~p opened~n", [Filename]),
        file:write(IoDevice, P),
	    file:close(IoDevice);
	error ->
	    io:format("error opening file~n", [])
    end.


x( {X,_Y} ) ->
    X;
x( {X,_Y,_Z} ) ->
    X.

y( {_X,Y} ) ->
    Y;
y( {_X,Y,_Z} ) ->
    Y.

z( {_X,_Y,Z} ) ->
    Z.

xz( {X,_Y,Z} ) ->
    {X, Z}.

xy( {X,Y,_Z} ) ->
    {X, Y}.

yz( {_X,Y,Z} ) ->
    {Y,Z}.

xyy( {X,Y} ) ->
    {X, Y, Y}.

xxx( {X,_Y} ) ->
    {X, X, X}.

yyx( {X,Y} ) ->
    {Y, Y, X}.

yxy( {X,Y} ) ->
    {Y, X, Y}.


pow(A,B) -> math:pow(A,B).

length2({X,Y,Z}) ->
    math:sqrt(pow(X,2) + pow(Y,2) + pow(Z,2));
length2({X,Y}) ->
    math:sqrt(pow(X,2) + pow(Y,2)).

length6({X,Y,Z}) ->
    math:pow(pow(X,6) + pow(Y,6) + pow(Z,6), 1.0/6.0);
length6({X,Y}) ->
    math:pow(pow(X,6) + pow(Y,6), 1.0/6.0).

length8({X,Y,Z}) ->
    math:pow(pow(X,8) + pow(Y,8) + pow(Z,8), 1.0/8.0);
length8({X,Y}) ->
    math:pow(pow(X,8) + pow(Y,8), 1.0/8.0).

mod( {X1, Y1, Z1}, {X2, Y2, Z2} )->
    { mod(X1,X2), mod(Y1,Y2), mod(Z1,Z2)};
mod( A, B ) ->
    A-B*floor(A/B).

vec3( A ) ->
    {A, A, A}.
vec2( A ) ->
    {A, A}.

vecsub( {X1, Y1, Z1}, {X2, Y2, Z2} ) ->
    {X1-X2, Y1-Y2, Z1-Z2};
vecsub( {X1, Y1}, {X2, Y2} ) ->
    {X1-X2, Y1-Y2};
vecsub( {X1, Y1, Z1}, A ) ->
    {X1-A, Y1-A, Z1-A};
vecsub( {X1, Y1}, A ) ->
    {X1-A, Y1-A}.
vecadd( {X1, Y1, Z1}, {X2, Y2, Z2} ) ->
    {X1+X2, Y1+Y2, Z1+Z2};
vecadd( {X1, Y1}, {X2, Y2} ) ->
    {X1+X2, Y1+Y2};
vecadd( {X1, Y1, Z1}, A ) ->
    {X1+A, Y1+A, Z1+A}.
vecadd( [] ) -> {0, 0, 0};
vecadd( [Head | Veclist] ) ->
    vecadd( Head, vecadd(Veclist)).
vecmult( {X1, Y1, Z1}, {X2, Y2, Z2} ) ->
    {X1*X2, Y1*Y2, Z1*Z2};
vecmult( {X1, Y1}, {X2, Y2} ) ->
    {X1*X2, Y1*Y2};
vecmult( {X1, Y1, Z1}, A ) ->
    {X1*A, Y1*A, Z1*A};
vecmult( {X1, Y1}, A ) ->
    {X1*A, Y1*A}.

vecdiv( {X1, Y1, Z1}, {X2, Y2, Z2} ) ->
    {X1/X2, Y1/Y2, Z1/Z2};
vecdiv( {X1, Y1}, {X2, Y2} ) ->
    {X1/X2, Y1/Y2};
vecdiv( {X1, Y1, Z1}, A ) ->
    {X1/A, Y1/A, Z1/A};
vecdiv( {X1, Y1}, A ) ->
    {X1/A, Y1/A}.

vecfunc( {X, Y, Z}, Func, B) when is_function(Func, 2) ->
    {Func(X, B), Func(Y, B), Func(Z, B)};
vecfunc( {X, Y}, Func, B) when is_function(Func, 2) ->
    {Func(X, B), Func(Y, B)}.
vecfunc( {X, Y, Z}, Func) when is_function(Func, 1) ->
    {Func(X), Func(Y), Func(Z)};
vecfunc( {X, Y}, Func) when is_function(Func, 1) ->
    {Func(X), Func(Y)}.

vecabs( P ) ->
    vecfunc( P, fun(A) -> abs(A) end).

vecmax( {X1, Y1, Z1}, {X2, Y2, Z2} ) ->
    {max(X1,X2), max(Y1,Y2), max(Z1,Z2)};
vecmax( P, Max ) ->
    vecfunc( P, fun(A, B) -> max(A, B) end, Max).

normalize( P = {X,Y,Z} ) ->
    Len = length2(P),
    %io:format("|~p| = ~p~n ", [P, Len]),
    {X/Len, Y/Len, Z/Len};
normalize( P = {X,Y} ) ->
    Len = length2(P),
    {X/Len, Y/Len}.

clamp( {X,Y,Z}, Min, Max) ->
    { clamp(X, Min, Max), clamp(Y, Min, Max), clamp(Z, Min, Max) };
clamp( {X,Y}, Min, Max) ->
    { clamp(X, Min, Max), clamp(Y, Min, Max) };
clamp( V, Min, Max) ->
    if V<Min -> Min;
        V>=Max -> Max;
        true -> V
    end.

smoothstep(Edge0, Edge1, X) ->
    T = clamp((X - Edge0) / (Edge1 - Edge0), 0.0, 1.0),
    T * T * (3.0 - 2.0 * T).

dot([A|As], [B|Bs]) -> (A*B) + dot(As, Bs);
dot([],	[])	-> 0;
dot({X1,Y1}, {X2,Y2}) -> dot([X1,Y1], [X2,Y2]);
dot({X1,Y1,Z1}, {X2,Y2,Z2}) -> dot([X1,Y1,Z1], [X2,Y2,Z2]).

mix( {X1, Y1, Z1}, {X2, Y2, Z2}, A) ->
    F = fun(Min, Max) -> Min * (1-A) + Max * A end,
    {F(X1,X2), F(Y1, Y2), F(Z1, Z2)}.

reflect (I, N) ->
    vecsub(I, vecmult(N, 2.0 * dot(N, I)) ).

% http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
checkersBox( {X,Y} ) ->
    % filter kernel
    %vec2 w = fwidth(p) + 0.001;
    %// analytical integral (box filter)
    %vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    %// xor pattern
    %return 0.5 - 0.5*i.x*i.y;                  
    abs((floor(X)+floor(Y)) rem 2).

calcSoftshadow( _Or, _Dir, _T, _Tmax, 0, Res) -> Res;
calcSoftshadow( _Or, _Dir, T, Tmax, _I, Res) 
    when Res<0.005; T>Tmax -> Res;
calcSoftshadow( Or, Dir, T, Tmax, I, Res) ->
		{T2, _M} = map( vecadd( Or , vecmult( Dir, T ) ) ),
        Res2 = min( Res, 8.0*T2/T ),
        calcSoftshadow(Or, Dir, T + clamp(T2, 0.02, 0.10) , Tmax, I-1, Res2).

calcSoftshadow( Or, Dir, Mint, Tmax ) ->
    Res = calcSoftshadow( Or, Dir, Mint, Tmax, 16, 1.0 ),
    clamp( Res, 0.0, 1.0 ).


%%------------------------------------------------------------------

%Substraction
opS( D1, D2 ) ->
    max(-D2,D1).

%Union
opU( D1 = {T1, _M1}, D2 = {T2, _M2} ) ->
	case T1<T2 of
        true -> D1;
        false -> D2
    end.
opU( [D] ) -> D;
opU( [D1 = {T1, _M1} | List] ) ->
    D2 = opU( List ),
    {T2, _M2} = D2,
	case T1<T2 of
        true -> D1;
        false -> D2
    end.

opRep( P, C ) ->
    vecsub(mod(P,C), vecmult(C,0.5)).

opTwist( {X,Y,Z} ) ->
    C = math:cos(10.0*Y+10.0),
    S = math:sin(10.0*Y+10.0),
    %m = mat2(C,-S,S,C),
    %return vec3(m*p.xz,p.y).
    {X*C-Z*S, X*S + Z*C, Y}.

%%------------------------------------------------------------------

sdPlane( P ) -> y(P).


sdSphere( P, S ) ->
    length2(P)-S.

sdBox( P, B ) ->
    D = {X,Y,Z} = vecsub(vecabs( P ), B), %vecfunc(P, fun(A) -> abs(A) end), B),
    Dmax = vecfunc(D, fun(A) -> max(A, 0.0) end),
    min(max(X,max(Y,Z)),0.0) + length2(Dmax).

sdEllipsoid( P, R={Xr, Yr, Zr} ) ->
    (length2( vecdiv( P, R ) ) - 1.0) * min(min(Xr, Yr),Zr).

udRoundBox( P, B, R ) ->
    length2(vecmax(vecsub(vecabs(P),B),0.0))-R.

sdTorus( P, T ) ->

    length2( { length2(xz(P))-x(T) ,y(P) } )-y(T).

sdHexPrism( P, H ) ->
    Q = vecabs(P),
    D1 = z(Q)-y(H),
    D2 = max((x(Q)*0.866025+y(Q)*0.5),y(Q))-x(H),
    length2(vecmax({D1,D2},0.0)) + min(max(D1,D2), 0.0).

sdCapsule( P, A, B, R ) ->
	PA = vecsub(P, A),
    BA = vecsub(B, A),
	H = clamp( dot(PA, BA)/dot(BA, BA), 0.0, 1.0 ),
	length2( vecsub(PA, vecmult(BA,H) ) ) - R.

sdEquilateralTriangle( P ) ->
    K = 1.73205, %sqrt(3.0)
    Px = abs(x(P)) - 1.0,
    Py = y(P) + 1.0/K,
    {P2x, P2y} = if
        Px + K*Py > 0.0 ->
            {(Px - K*Py)/2, (-K*Px - Py)/2};
        true ->
            {Px, Py}
        end,
    -length2({P2x + 2.0 - clamp( ( P2x + 2.0 ) / 2.0, 0.0, 1.0),
             P2y}
            ) * (P2y/abs(P2y)).

sdTriPrism( P, H ) ->
    Q = vecabs(P),
    D1 = z(Q)-y(H),
    % distance bound
    D2 = max((x(Q)*0.866025+y(P)*0.5),-y(P))-x(H)*0.5,
    length2(vecmax({D1,D2},0.0)) + min(max(D1,D2), 0.0).

sdCylinder( P, H ) ->
  D = vecsub(vecabs({length2(xz(P)),y(P)}), H),
  min(max(x(D),y(D)),0.0) + length2(vecmax(D,0.0)).

sdCone( P, C ) ->
    Q = { length2(xz(P)), y(P) },
    D1 = -y(Q)-z(C),
    D2 = max( dot(Q,xy(C)), y(Q)),
    length2(vecmax({D1, D2},0.0)) + min(max(D1, D2), 0.0).

sdConeSection( P, H, R1, R2 ) ->
    D1 = -y(P) - H,
    Q = y(P) - H,
    Si = 0.5*(R1-R2)/H,
    D2 = max( math:sqrt( dot(xz(P),xz(P))*(1.0-Si*Si)) + Q*Si - R2, Q ),
    length2(vecmax({D1,D2},0.0)) +min(max(D1,D2), 0.0).

sdPryamid4( P, {CosA, SinA, Height} ) -> % h = { cos a, sin a, height }
    % Tetrahedron = Octahedron - Cube
    Box = sdBox( vecsub(P , {0,-2.0*Height,0}), vec3(2.0*Height) ),
 
    D1 = max( 0.0, abs( dot(P, { -CosA, SinA, 0 }) )),
    D2 = max( D1, abs( dot(P, { CosA, SinA, 0 }) )),
    D3 = max( D2, abs( dot(P, { 0, SinA, CosA }) )),
    D4 = max( D3, abs( dot(P, { 0, SinA,-CosA}) )),
    Octa = D4 - Height,
    % Subtraction
    max(-Box,Octa).

sdTorus82( P, T ) ->
    Q = {length2(xz(P))-x(T), y(P)},
    length8(Q)-y(T).


sdTorus88( P, T ) ->
    Q = {length8(xz(P))-x(T), y(P)},
    length8(Q)-y(T).

sdCylinder6( P, H ) ->
    max( length6(xz(P))-x(H), abs(y(P))-y(H) ).

map( Pos ) ->
    opU( [
        {sdPlane(           Pos ), 1.0},
        {sdSphere(  vecsub( Pos, { 0.0, 0.25, 0.0}), 0.25 ),                    46.9},
        {sdBox(     vecsub( Pos, { 1.0, 0.25, 0.0}), vec3(0.25) ),               3.0},
        {udRoundBox( vecsub(Pos, { 1.0, 0.25, 1.0}), vec3(0.15),        0.1 ),  41.0},
        {sdTorus(   vecsub( Pos, { 1.0, 0.25, 1.0}), {0.20, 0.05}),             25.0},
        {sdCapsule(         Pos, {-1.3, 0.10,-0.1},  {-0.8, 0.50, 0.2}, 0.1 ),  31.9},
        {sdTriPrism( vecsub(Pos, {-1.0, 0.25,-1.0}), {0.25, 0.05}),             43.5},
        {sdCylinder( vecsub(Pos, { 1.0, 0.30,-1.0}), {0.1,0.2}),                 8.0},
        {sdCone(     vecsub(Pos, { 0.0, 0.50,-1.0}), {0.8,0.6,0.3}),            55.0},
        {sdTorus82(  vecsub(Pos, { 0.0, 0.25, 2.0}), {0.20,0.05}),              50.0},
        {sdTorus88(  vecsub(Pos, {-1.0, 0.25, 2.0}), {0.20,0.05}),              43.0},
        {sdCylinder6(vecsub(Pos, { 1.0, 0.30, 2.0}), {0.1,0.2}),                12.0},
        {sdHexPrism( vecsub(Pos, {-1.0, 0.20, 1.0}), {0.25,0.05}),              17.0},
        {sdPryamid4( vecsub(Pos, {-1.0, 0.15,-2.0}), {0.8,0.6,0.25}),           37.0},
        {opS(udRoundBox(vecsub(Pos, {-2.0,0.2, 1.0}), vec3(0.15),      0.05),
             sdSphere  (vecsub(Pos, {-2.0,0.2, 1.0}), 0.25)),                   13.0},
        {opS(sdTorus82 (vecsub(Pos, {-2.0,0.2, 0.0}), {0.20,0.1}),
             sdCylinder( opRep( {math:atan(x(Pos)+2.0 / z(Pos)/6.2831),
                                 y(Pos),
                                 0.02+0.5*length2(vecsub(Pos, {-2.0, 0.2, 0.0}))
                                }, {0.05,1.0,0.05}
                              ), {0.02, 0.6})),                                 51.0},
        {0.5*sdSphere( vecsub(Pos, {-2.0, 0.25,-1.0}), 0.2 ) +
             0.03 * math:sin(50.0*x(Pos)) *
             math:sin(50.0*y(Pos)) * 
             math:sin(50.0*z(Pos)),                                             65.0},
        {0.5*sdTorus( opTwist( vecsub(Pos, {-2.0,0.25, 2.0})), {0.20,0.05} ),   46.7},
        {sdConeSection(vecsub(Pos, { 0.0, 0.35,-2.0}), 0.15, 0.2, 0.1),         43.17},
        {sdEllipsoid(  vecsub(Pos, { 1.0, 0.35,-2.0}), {0.15, 0.2, 0.05} ),     43.17}
        ]
    ).
    %{sdPlane( Pos ), 1.0}.