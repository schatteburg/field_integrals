import numpy as np
from scipy.integrate import quad
from abc import ABC

class field():

    def __init__(self, fieldvalues: np.ndarray, var: list[np.ndarray], coord_sys: str):
        # if not (type(fieldvalues) == np.ndarray and type(var)==)

        # check number of dimensions match between inputs
        if not len(fieldvalues.shape) == len(var):
            raise ValueError(f"Number of dimensions of inputs don't match:\nlen(fieldvalues.shape) = {len(fieldvalues.shape)};\tlen(var) = {len(var)}.")
        else:
            self.ndims = len(var)
        
        # check number of points per dimension match between inputs
        if not fieldvalues.shape == tuple(v.size for v in var):
            raise ValueError(f"Number of points per dimension don't match between input variables.")
        else:
            self.shape = fieldvalues.shape
        
        # check number of dimensions and coordinate system are supported
        if self.ndims == 1 and coord_sys not in ["cartesian", "polar"]  or self.ndims == 2 and coord_sys not in ["cartesian", "polar"] or self.ndims == 3 and coord_sys not in ["cartesian", "cylindrical"] or self.ndims > 3 or self.ndims <=0:
            raise ValueError(f'Number of dimensions ({self.ndims}) and/or coordinate system ({coord_sys}) are not supported.')

        self.values = fieldvalues
        self.var = var
        self.cs = coord_sys
    
    def integrate_all_dimensions(self):
        if self.ndims == 1:
            print('integrating dimension 0')
            if self.cs == "cartesian":
                return np.trapz(self.values, x=self.var[0], axis=0)
            elif self.cs in ["polar", "cylindrical"]:
                # raise NotImplementedError(f"Coordinate system {self.cs} is not implemented.")
                return np.trapz(self.var[0]*self.values, x=self.var[0], axis=0)
        else:
            print(f'integrating dimension {self.ndims-1}')
            newvalues = np.trapz(self.values, x=self.var[-1], axis=self.ndims-1)
            if self.ndims == 3 and self.cs == "cylindrical":
                newcs = "polar"
            else:
                newcs = self.cs
            return field(newvalues, self.var[:-1], newcs).integrate_all_dimensions() # recursive call