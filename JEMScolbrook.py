#SOURCE: https://www.damtp.cam.ac.uk/user/mjc249/pdfs/JEMS_foundations_colbrook.pdf The foundations of spectral computations via the Solvability Complexity Index hierarchy by Matthew J. Colbrook, Anders C. Hansen (2022)

from fractions import Fraction
from collections.abc import Callable
from math import isqrt
import numpy as np
class JEMScolbrook:
    '''
    Implementation of algorithms from "The foundations of spectral 
       computations via the Solvability Complexity Index hierarchy"
       by Colbrook & Hansen (2022)
    '''

    # CompInvg
    
    def _type_validation_compInvg(self, n: int, y: float, g:Callable[[float], float]) -> None:
        if not isinstance(n, int): 
            raise TypeError("n must be an int and not float") 
        if n <= 0:
            raise ValueError("n must be positive")  
        if y < 0:
            raise ValueError("y must be non-negative") 
        if abs(g(0.0)) > 1e-10:
            raise ValueError("We must have g(0) = 0, g(0) falls out of floating point tolerance.")
    def CompInvg_slow(self, n: int, y: float, g:Callable[[float], float], max_iter = 10**7, init_guess = 0) -> Fraction:
        '''
        Approximate g^(-1)(y) using a discrete mesh of size 1/n. Specifically, we find the least k such that g(k/n) > y and hence give an approximation to g^(-1)(y) to precision 1/n. 

        _slow: Brute force method, does not use narrowing window search method (e.g. binary search). 
        
        Parameters
        -------------
        n : int 
            size of mesh, must satisfy n > 0
        y : float 
            input for which we want to approximate g^(-1)(y)
        g : collections.abc.Callable[[float], float]
            increasing function g : R_+ -> R_+ representing resolvent control. Must satisfy g(0) = 0, g(x) <= x and be monotone increasing. That g(x) <= x or g is monotone is not checked.
        max_iter : int 
            maximum number of iterations to find k/n before termination
        init_guess : int 
            initial guess for k.
        
        Returns 
        -------------
        Fraction 
            rational approximation k/n of g^(-1)(y)
        
        Raises
        -------------
        ValueError 
            Occurs if:
                n <= 0 
                y < 0
                g(0) != 0
        TypeError 
            if n is not an integer.
        RuntimeError 
            if a suitable approximation is not found by max_iter iterations.
        
        Big-O Complexity 
        -------------
        O(n) assuming inexpensive g, does n iterations
        '''
        # input validation 
        self._type_validation_compInvg(n, y, g)
        # We first identify a 1-wide interval within which g first exceeds y
        j = init_guess
        while g(j + 1) <= y:
            j += 1
            if j == max_iter:
                raise RuntimeError(f"max_iter ({max_iter}) exceeded")
        # once we've exited this loop, we know g(j + 1) > y and g(j) <= y. Hence g^(-1)(y) \in [j, j + 1) = [(j*n)/n, (j*(n + 1))/n)
        for k in range(j*n, (j + 1)*n):
            if g(k/n) > y:
                return Fraction(k, n) # using fraction to avoid floating point errors
    
    #DistSpec 
    def DistSpec_slow(self, matrix:Callable[[int, int], complex], n:int, z:complex, f:Callable[[int], int], max_iter = 10**7) -> Fraction:
        '''
        Approximate norm(R(z, A))^(-1) with mesh size 1/n given dispersion f
        
        _slow: Brute force method, checks each l individually and computes all eigenvalues before concluding on positive definiteness.
        
        Parameters 
        -------------
        matrix : collections.abc.Callable 
            function N^2 -> C representing a closed infinite matrix. 
        
        In the following, A = (matrix(i, j))
        n : int 
            size of mesh, must satisfy n > 0
        z : complex 
            z in norm(R(z, A))^(-1)
        f : collections.abc.Callable[[int], int]
            a dispersion control for the matrix, accepting ints and giving ints
        max_iter : int 
            maximum number of iterations to find k/n before termination
        
        Returns
        -------------
        Fraction
            rational approximation for norm(R(z, A))^(-1)
        
        Raises 
        -------------
        TypeError
            if f(n) is not an integer
        ValueError 
            if f(n) does not satisfy f(n) >= n: f is not a valid dispersion bound 
        RuntimeError
            if a suitable approximation is not found in max_iter iterations
        
        Big-O Complexity 
        -------------
        O(f(n)*n^2 + max_iter*n^3) - since max_iter dominates f(n) this will scale more like O(max_iter*n^3). 
        
        first term: O(f(n)*n) for building the matrices, O(f(n)*n^2) from matrix multiplication.
        
        second term: eigenvalue search is O(n^3), we do this up to max_iter times.
        '''
        fn = f(n) # pre-compute f(n) in case it is expensive
        if not (isinstance(fn, int)): #check if f(n) is an integer 
            raise TypeError(f"f(n) ({fn}) is not an int")
        if not fn >= n: # check if f(n) >= n
            raise ValueError(f"f(n) ({fn}) is not >= n")
        B = np.array([[matrix(i, j) - z for j in range(n)] for i in range(fn)]) # (A - z I)(1 : f(n))(1 : n)
        C = np.array([[matrix(j, i).conjugate() - z.conjugate() for j in range(n)] for i in range(fn)]) # (A - z I)*(1 : f(n))(1 : n)
        S = np.matmul(np.conjugate(B).T, B) # S = B*B
        S_size = S.shape[0] # get size of S to identify suitable identity matrix 
        id_S = np.identity(S_size)
        T = np.matmul(np.conjugate(C).T, C) # T = C*C
        T_size = T.shape[0] # get size of T to identify suitable identity matrix 
        id_T = np.identity(T_size)
        v = True
        l = 1
        while v and l < max_iter:
            l += 1
            l2 = l*l  
            n2 = n*n
            p = np.all(np.linalg.eigvalsh(S - (l2/n2)*id_S) > 0) # check whether S - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
            q = np.all(np.linalg.eigvalsh(T - (l2/n2)*id_T) > 0) # check whether T - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
            v = p and q
        if l == max_iter:
            raise RuntimeError(f"max_iter ({max_iter}) exceeded")
        return Fraction(l, n) # using fraction to avoid floating point errors
    
    # grid generation 
    def _generate_grid_input_val(self, n:int) -> None:
        if not isinstance(n, int):
            raise TypeError("n is not an int")
        if n <= 0:
            raise ValueError("n is non-positive")
    def generate_grid_slow(self, n:int) -> list[complex]:
        '''
        Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes . 
        
        _slow: Brute force method checking each candidate (x, y) individually. 
        
        Parameters
        -------------
        n : int 
            mesh size for grid
        
        Returns 
        -------------
        list[complex]
            List of complex numbers corresponding to Grid(n) 
        
        Raises 
        -------------
        TypeError: 
            if n is not an integer
        ValueError:
            if n <= 0
        Big-O Complexity 
        -------------
        O(n^4) - n^2 values of x, and n^2 values of y for each x. 
        '''
        # input validation
        self._generate_grid_input_val(n)
        # return 
        return [complex(x, y)/n for x in range(-n*n, n*n + 1) for y in range(-n*n, n*n + 1) if x*x + y*y <= n**4]
    def generate_grid(self, n:int) -> list[complex]:
        '''
        Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes.
        
        not _slow: Given an x, (x, y) \in Grid(n) if and only if y^2 <= n^4 - x^2. That is, if and only if |y| <= floor(sqrt(n^4 - x^2)) := y_max. Hence we enumerate up to this y_max.
        
        Parameters
        -------------
        n : int 
            mesh size for grid
        
        Returns 
        -------------
        list[complex]
            List of complex numbers corresponding to Grid(n) 
        
        Raises 
        -------------
        TypeError: 
            if n is not an integer
        ValueError:
            if n <= 0
        
        Big-O Complexity 
        -------------
        O(n^4), but slightly (typically performs ~79% as many calcluations) faster because of narrowed search range. 
        '''
        # input validation 
        self._generate_grid_input_val(n)
        #method 
        grid = [] #init empty grid
        r = n*n #pre-compute r^2 so we don't have to re-compute it every loop
        r_squared = r*r # pre-compute r^4 to avoid re-computation
        for x in range(-r, r + 1): #iterate over x
            #as in docstring, y_max cannot be bigger than this and moreover every y in this range corresponds to a point in Grid(n) 
            max_y = isqrt(r_squared - x*x) 
            for y in range(-max_y, max_y + 1):
                grid.append(complex(x/n, y/n))
        return grid
