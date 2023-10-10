import numpy as np
from scipy.integrate import quad
from abc import ABC


######### decorators #########
def checked_field_compatibility(func: callable) -> callable:
    def wrapper(self, other: "field", *args, **kwargs):
        if not self.cs == other.cs:
            raise ValueError(f"Coordinate systems of fields don't match: {self.cs} and {other.cs}.")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper



class field():

    def __init__(self, fieldvalues: np.ndarray, var: list[np.ndarray], coord_sys: str) -> None:
        # if not (type(fieldvalues) == np.ndarray and type(var)==)

        # check whether number of dimensions match between inputs
        if not len(fieldvalues.shape) == len(var):
            raise ValueError(f"Number of dimensions of inputs don't match:\nlen(fieldvalues.shape) = {len(fieldvalues.shape)};\tlen(var) = {len(var)}.")
        else:
            self.ndims = len(var)
        
        # check whether number of points per dimension match between inputs
        if not fieldvalues.shape == tuple(v.size for v in var):
            raise ValueError(f"Number of points per dimension don't match between input variables.")
        else:
            self.shape = fieldvalues.shape
        
        # check whether number of dimensions and coordinate system are supported
        if self.ndims == 1 and coord_sys not in ["cartesian", "polar"]  or self.ndims == 2 and coord_sys not in ["cartesian", "polar"] or self.ndims == 3 and coord_sys not in ["cartesian", "cylindrical"] or self.ndims > 3 or self.ndims <=0:
            raise ValueError(f'Number of dimensions ({self.ndims}) and/or coordinate system ({coord_sys}) are not supported.')

        self.values = fieldvalues
        self.var = var
        self.cs = coord_sys
    
    def integrate_all_dimensions(self, vocal: bool = False) -> float:
        if self.ndims == 1:
            if vocal:
                print('integrating dimension 0')
            if self.cs == "cartesian":
                return np.trapz(self.values, x=self.var[0], axis=0)
            elif self.cs in ["polar", "cylindrical"]:
                # raise NotImplementedError(f"Coordinate system {self.cs} is not implemented.")
                return np.trapz(self.var[0]*self.values, x=self.var[0], axis=0)
        else:
            if vocal:
                print(f'integrating dimension {self.ndims-1}')
            newvalues = np.trapz(self.values, x=self.var[-1], axis=self.ndims-1)
            if self.ndims == 3 and self.cs == "cylindrical":
                newcs = "polar"
            else:
                newcs = self.cs
            return field(newvalues, self.var[:-1], newcs).integrate_all_dimensions(vocal=vocal) # recursive call
    
    def normalize(self, vocal: bool = False)-> "field":
        return field(self.values/self.integrate_all_dimensions(vocal=vocal), self.var, self.cs)
    
    def normalize_abs2(self, vocal: bool = False) -> "field":
        N = field(self.values*self.values.conj(), self.var, self.cs).integrate_all_dimensions(vocal=vocal)
        # N = (self*self.conj()).integrate_all_dimensions()
        return field(self.values/np.sqrt(N), self.var, self.cs)
    
    @checked_field_compatibility
    def overlap(self, other: "field", vocal: bool = False) -> float:
        return (self*other.conj()).integrate_all_dimensions(vocal=vocal)
    
    def conj(self) -> "field":
        return field(self.values.conj(), self.var, self.cs)

    ######### magic methods #########
    # unary operators    
    def __repr__(self) -> str:
        return f"field in {self.ndims}D, {self.cs} coordinate system, with shape {self.shape}"
    
    def __abs__(self) -> "field":
        return field(np.abs(self.values), self.var, self.cs)
    
    def __pow__(self, power: float) -> "field":
        return field(self.values**power, self.var, self.cs)
    
    # binary operators
    @checked_field_compatibility
    def __eq__(self, other: "field") -> bool:
        return np.all(self.values == other.values) and np.all(self.var == other.var) and self.cs == other.cs
    
    @checked_field_compatibility
    def __add__(self, other: "field") -> "field":
        return field(self.values+other.values, self.var, self.cs)
    
    @checked_field_compatibility
    def __sub__(self, other: "field") -> "field":
        return field(self.values-other.values, self.var, self.cs)
    
    @checked_field_compatibility
    def __mul__(self, other: "field") -> "field":
        return field(self.values*other.values, self.var, self.cs)

    @checked_field_compatibility
    def __truediv__(self, other: "field") -> "field":
        return field(self.values/other.values, self.var, self.cs)


    
    