from copy import deepcopy


class DeepCopyMixin:
    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''
        return deepcopy(self)

class GetMixin:
    def get(self, key):
        '''
        Function to retrieve and return an attribute of the current object.
        '''
        return getattr(self, key)

    def __getitem__(self, key):
        '''
        Function to retrieve and return an attribute of the current object.
        '''
        return getattr(self, key)
