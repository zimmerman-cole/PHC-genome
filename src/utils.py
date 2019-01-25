"""
Utility functions.
"""

def is_iterable(obj):
    try:
        list(obj)
        return True
    except TypeError:
        return False
   