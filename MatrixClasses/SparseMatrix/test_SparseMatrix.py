import pytest
from SparseMatrix import SparseMatrix 
from DiagonalMatrix import DiagonalMatrix
from fractions import Fraction

class TestSparseMatrixInit:
    def test_init_zero_matrix(self):
        zero_entries = lambda i, j : 0
        zero_f = lambda i : 0
        zero_matrix = SparseMatrix(zero_entries, zero_f) 
        
        assert (zero_matrix.entries is zero_entries) and (zero_matrix.f is zero_f) and (zero_matrix.tolerance == 1e-16)

    def test_check_init_tolerance_validation_int(self):
        zero_entries = lambda i, j : 0
        zero_f = lambda i : 0
        with pytest.raises(TypeError, match = "Tolerance must be float or Fraction"):
            matrix = SparseMatrix(zero_entries, zero_f, 133)

    def test_check_init_tolerance_validation_str(self):
        zero_entries = lambda i, j : 0
        zero_f = lambda i : 0
        with pytest.raises(TypeError, match = "Tolerance must be float or Fraction"):
            matrix = SparseMatrix(zero_entries, zero_f, 'Dog')

class TestSparseMatrixCallable:
    def test_call_identity_matrix(self):
        id_entries = lambda i, j : int(i == j)
        id_f = lambda i : i
        id_matrix = SparseMatrix(id_entries, id_f) 
            
        assert id_matrix(129084098235923, 129084098235922) == 0
        assert id_matrix(134254353453, 134254353453) == 1
    
    def test_call_banded_matrix(self):
        banded_entries = lambda i, j : i - j
        banded_f = lambda i : i + 2 # hence a_ij != 0 implies |i - j| <= 2
        banded_matrix = SparseMatrix(banded_entries, banded_f)
        
        assert banded_matrix(10, 3) == 0
        assert banded_matrix(10, 12) == -2
        assert banded_matrix(11, 10) == 1
    
    def test_f_check(self):
        entries = lambda i, j : i * j # note that this is non-zero if i, j > 10
        f = lambda i : 10 
        matrix = SparseMatrix(entries, f)
        
        assert SparseMatrix(15, 10) == 0
        assert SparseMatrix(10, 15) == 0
        assert SparseMatrix(7, 7) == 49

class TestSparseMatrixAddition:
    def test_addition_with_zero(self):
        banded_entries = lambda i, j : i - j
        banded_f = lambda i : i + 2
        banded_matrix = SparseMatrix(banded_entries, banded_f)
        
        zero_entries = lambda i, j : 0
        zero_f = lambda i : 0
        zero_matrix = SparseMatrix(zero_entries, zero_f) 
        
        result = banded_matrix + zero_matrix 
        
        assert result(10000000, 10000001) == banded_matrix(10000000, 10000001)
        assert result(9999999, 10000000) == banded_matrix(9999999, 10000000)
        assert result(1439, 349024) == banded_matrix(1439, 349024)
    
    def test_addition_with_one_nonzero(self):
        banded_entries = lambda i, j : i - j
        banded_f = lambda i : i + 2
        banded_matrix = SparseMatrix(banded_entries, banded_f)
        
        all_but_one = lambda i, j : 1 if (i, j) == (1, 2) else 0
        abo_f = lambda i : 2
        abo = SparseMatrix(all_but_one, abo_f) 
        
        result = abo + banded_matrix
        
        assert result(1, 2) == abo(1, 2) + 1
        assert result(3, 4) == abo(3, 4)
    
    def test_addition_commutativity(self):
        banded_entries_1 = lambda i, j : i - j if abs(i - j) <= 2 else 0
        banded_f_1 = lambda i : i + 2 
        banded_matrix_1 = SparseMatrix(banded_entries_1, banded_f_1)
        
        banded_entries_2 = lambda i, j : i + j
        banded_f_2 = lambda i : i + 52
        banded_matrix_2 = SparseMatrix(banded_entries_2, banded_f_2) 
        
        left_sum = banded_matrix_1 + banded_matrix_2
        right_sum = banded_matrix_2 + banded_matrix_1 
        
        assert left_sum(50, 102) == right_sum(50, 102)
        assert left_sum(532, 533) == right_sum(532, 533)
        assert left_sum(0, 0) == right_sum(0, 0)
    
    def test_addition_type_check(self):
        zero_entries = lambda i, j : 0
        zero_f = lambda i : 0
        zero_matrix = SparseMatrix(zero_entries, zero_f)
        
        with pytest.raises(TypeError, match = "Cannot add SparseMatrix with object not of SparseMatrix type."):
            matrix1 = zero_matrix + 1
        
        with pytest.raises(TypeError, match = "Cannot add SparseMatrix with object not of SparseMatrix type."):
            matrix2 = zero_matrix + [1, 2, 3]
        
        with pytest.raises(TypeError, match = "Cannot add SparseMatrix with object not of SparseMatrix type."):
            matrix3 = zero_matrix + 'foobar'
            
class TestSparseMatrixRMultiplication:
    def test_rmultiplication_scalar_type_check(self):
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f) 
        
        with pytest.raises(TypeError, match = "Cannot multiply SparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction"):
            matrix1 = identity_matrix.__rmul__('foobar') 
        
        with pytest.raises(TypeError, match = "Cannot multiply SparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction"):
            matrix2 = identity_matrix.__rmul__([1, 2, 3]) 
    
    def test_rmultiplication_by_almost_zero_scalar(self):
        tol = 1e-16
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f, tol)
        
        zero_matrix = identity_matrix.__rmul__(1e-17)
        
        assert zero_matrix(0, 0) == 0
        assert zero_matrix(1, 2) == 0
        assert zero_matrix(143092, 143092) == 0
    
    def test_rmultiplication_by_nonzero_scalar(self):
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f) 
        
        float_matrix = identity_matrix.__rmul__(3.0)
        int_matrix = identity_matrix.__rmul__(3)
        complex_matrix = identity_matrix.__rmul__(3+2j)
        fraction_matrix = identity_matrix.__rmul__(Fraction(4, 5))
        
        assert float_matrix(55, 55) == 3.0
        assert float_matrix(4, 2) == 0.0
        
        assert int_matrix(55, 55) == 3
        assert int_matrix(4, 2) == 0
        
        assert complex_matrix(55, 55) == 3+2j
        assert complex_matrix(4, 2) == 0
        
        assert fraction_matrix(55, 55) == Fraction(4, 5)
        assert fraction_matrix(4, 2) == 0
        

class TestSparseMatrixMultiplication:
    def test_multiplication_scalar_type_check(self):
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f) 
        
        with pytest.raises(TypeError, match = "Cannot multiply SparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction"):
            matrix1 = identity_matrix.__mul__('foobar') 
        
        with pytest.raises(TypeError, match = "Cannot multiply SparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction"):
            matrix2 = identity_matrix.__mul__([1, 2, 3]) 
    
    def test_multiplication_by_almost_zero_scalar(self):
        tol = 1e-16
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f, tol)
        
        zero_matrix = identity_matrix.__mul__(1e-17)
        
        assert zero_matrix(0, 0) == 0
        assert zero_matrix(1, 2) == 0
        assert zero_matrix(143092, 143092) == 0
    
    def test_rmultiplication_by_nonzero_scalar(self):
        identity_entries = lambda i, j : (i == j)
        identity_f = lambda i : i 
        identity_matrix = SparseMatrix(identity_entries, identity_f) 
        
        float_matrix = identity_matrix.__mul__(3.0)
        int_matrix = identity_matrix.__mul__(3)
        complex_matrix = identity_matrix.__mul__(3+2j)
        fraction_matrix = identity_matrix.__mul__(Fraction(4, 5))
        
        assert float_matrix(55, 55) == 3.0
        assert float_matrix(4, 2) == 0.0
        
        assert int_matrix(55, 55) == 3
        assert int_matrix(4, 2) == 0
        
        assert complex_matrix(55, 55) == 3+2j
        assert complex_matrix(4, 2) == 0
        
        assert fraction_matrix(55, 55) == Fraction(4, 5)
        assert fraction_matrix(4, 2) == 0

class TestSparseMatrixMatMultiplication:
    pass

class TestSparseMatrixSubtraction:
    pass

class TestSparseMatrixTranspose:
    pass

class TestSparseMatrixGetters:
    pass

class TestSparseMatrixSetters:
    pass
