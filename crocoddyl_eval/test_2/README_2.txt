quadruped_walkgen.ActionModelQuadrupedStepTime()

The global state : Y = [ X , P , dt ], nx = 21
X --> position, rotation angles, lin vel, rot vel x12
P --> Position of the feet in X,Y plan , x8
dt --> integration time, x1

The command : U = [dx1,dy1,dx2,dy2] , nu = 4 , 

cost = 0.5*d1|| X - X_ref ||^2 + 0.5*d2|| P - P_ref ||^2 + 0.5*d3|| dt - dt_ref||^2
      + 0.5*d4||u||^2 
	+ 0.5*d5||(dt_min - dt)^+||^2 + 0.5*d5||(dt - dt_max)^+||^2 + 0.5*d6(u0**2 + u1**2 - beta*dt**2)^+ + 0.5*d6(u2**2 + u3**2 - beta*dt**2)^+ 

Warning : d4 and d6 are not supposed to be used at the same time. The first one is to optimise around the shoulder position, by minimizing the norm of the jump ||u||^2 between the shoulder position and the final position. It is associated with d2 in the next model to keep the position around a certain pref. --> To use when not optimising the period (beginning of the gait).

d6 is used when optimizing the period to reduce the jump if the previous period cycle was fast and vice versa. 

########################################################################

quadruped_walkgen.ActionModelQuadrupedTime()
The global state : Y = [ X , P , dt ], nx = 21
X --> position, rotation angles, lin vel, rot vel x12
P --> Position of the feet in X,Y plan , x8
dt --> integration time, x1

The command : U = [dt] , nu = 1, 
X+1(21) = u

cost = 0.5*d1|| X - X_ref ||^2 + 0.5*d2|| P - P_ref ||^2  + 0.5*d3|| dt - dt_ref||^2 
      + 0.5*d3|| u - dt_ref||^2 +  0.5*d5||(dt_min - dt)^+||^2 + 0.5*d5||(dt - dt_max)^+||^2 
      + 0.5*d5||(dt_min - u)^+||^2 + 0.5*d5||(u - dt_max)^+||^2
