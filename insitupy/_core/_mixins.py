from copy import deepcopy

class DeepCopyMixin:
    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''
        # remove viewer instances
        
        return deepcopy(self)
