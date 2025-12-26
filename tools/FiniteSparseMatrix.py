class FiniteSparseMatrix:
    def __init__(self, entries : dict[tuple[int, int], float], default : float = 0.0):
        self.entries = entries
        self.default = default 
        
    def __call__(self, i : int, j : int) -> float:
        return self.entries.get((i, j), self.default)
    
    def __add__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        new_default = self.default + B.default
        self_size = len(self.entries)
        B_size = len(B.entries)
        smaller, bigger = self, B if self_size <= B_size else B, self
        new_entries = bigger.entries.copy()
        for idx in smaller.entries:
            candidate = new_entries.get(idx, 0.0) + smaller.entries[idx]
            if abs(candidate - new_default) < 1e-16:
                new_entries.pop(idx, None)
            else:
                new_entries[idx] = candidate                
        return FiniteSparseMatrix(new_entries, new_default)

    def __mul__(self, c : float) -> 'FiniteSparseMatrix':
        if c < 1e-16:
            return FiniteSparseMatrix({}, 0.0)
        else:
            new_entries = {key : c*val for key, val in self.entries.items()}
            new_default = c*self.default
            return FiniteSparseMatrix(new_entries, new_default)

    def __sub__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        return A + (-1)*B

    def __repr__(self) -> str:
        return f"FiniteSparseMatrix({len(self.entries)} entries, {self.default})"
       
    def get_entries(self) -> dict[tuple[int, int], float]:
        return self.entries 
    
    def set_entries(self, new_entries : dict[tuple[int, int], float]) -> None:
        for idx, val in new_entries.items():
            self.entries[idx] = val

    def pop(self, pos : dict[tuple[int, int], float]) -> None:
        self.entries.pop(pos, None)
    
    def get_default(self):
        return self.default
    
    def set_default(self, new_default : float) -> None:
        self.default = new_default

    
