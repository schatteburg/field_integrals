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
        if self.ndims == 1 and coord_sys not in ["cartesian", "polar"]  or self.ndims == 2 and coord_sys not in ["cartesian", "polar"] or self.ndims >= 3 or self.ndims <=0:
            raise ValueError(f'Number of dimensions ({self.ndims}) and/or coordinate system ({coord_sys}) are not supported.')

        self.values = fieldvalues
        self.var = var
        self.cs = coord_sys
    
    def integrate_area(self, afield = None):
        if afield is None:
            return self.integrate_area(afield=self)
        
        if afield.ndims == 1:
            if afield.cs == "cartesian":
                return np.trapz(afield.values, x=afield.var[0])
            elif afield.cs == "polar":
                raise NotImplementedError(f"Coordinate system {afield.cs} is not implemented.")
        else:
            return self.integrate_area(field(afield.values, afield.var[::-2], afield.cs)) # TODO: does not work to create new field with mismatched dimensions