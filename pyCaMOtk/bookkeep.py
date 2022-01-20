from copy import deepcopy
from numpy import argmin
from numpy.linalg import norm

from pyCaMOtk.check import is_type

class Database(object):
    """
    Database object maps a parameter (mu) to any number of targets.
    Useful in tracking parameters that have been visited and in what
    context.

    Properties
    ----------
    dist : function, 2 input arguments
      Distance function to compute distance between parameters

    mu_db : list of parameters (iterable of number)
      (Ordered) parameters stored in database

    target : dict of list of targets (iterable of number)
      Targets computed for each parameter; each entry in target
      has same length as mu_db, e.g., len(target[s]) == len(mu_db),
      and target[s][k] is the target corresponding to parameter mu_db[k].

    offset : int
      Offset to apply to index of entries in database

    copy : function, 1 input argument
      Copy memory of parameter or target

    Example
    -------
    >> dist = lambda a, b: norm([a[k]-b[k] for k in range(len(a))])
    >> db = Database(dist=dist, targets=['prim', 'dual'])
    >> db.add([0, 0], prim=[0], dual=[0])
    >> db.add([1, 0], prim=[2], dual=[0])
    >> db.add([0, 1], prim=[1], dual=[1])
    # db.mu_db = [[0, 0], [1, 0], [0, 1]]
    # db.target = {'prim': [[0], [2], [1]], 'dual': [[0], [0], [1]]}
    """
    def __init__(self, dist=None, targets=[], offset=0, copy=None):
        """
        Constructor

        Input arguments
        ---------------
        dist, offset : See above

        targets : iterable of str
          Name of all targets
        """
        if not is_type(targets, 'iter_of_str'):
            raise TypeError('targets must be iterable of str')
        if len(targets) == 0:
            raise ValueError('targets must have length > 0')
        if not is_type(offset, 'int'):
            if offset < 1:
                raise TypeError('offset must be non-negative int')
        if dist is None: dist = lambda a,b : norm(a-b)
        if copy is None: copy = deepcopy
        self.target = dict([(x, []) for x in targets])
        self.mu_db, self.dist, self.offset, self.copy = [], dist, offset, copy

    def check_permissible_target(self, wtarg):
        """
        Utility to check if target is permissible

        Input arguments
        ---------------
        wtarg : str
          Target name to check
        
        Return values
        -------------
        None
                
        Example
        -------
        >> db.check_permissible_target('prim')
        """
        if not is_type(wtarg, 'str'):
            raise TypeError('wtarg must be str')
        targs = [key for key in self.target]
        if wtarg not in targs:
            raise ValueError('wtarg must be: '+', '.join(targs))

    def add(self, mu, **wtarg):
        """
        Add parameter and corresponding targets to database. Robust to
        case that mu has already been seen.

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to add to database

        wtarg : dict with entries of iterable of number
          Targets to add to database corresponding to parameter mu

        Return value
        ------------
        idx : int
          Index of parameter mu in database

        Example
        -------
        >> db_sz = db.add([0, 0], prim=[0], dual=[0])
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        for key in wtarg:
            if not is_type(wtarg[key], 'iter_of_number'):
                raise TypeError('All targets must be iterable of number')

        # Handle case where mu already seen
        copy = self.copy
        idx, _ = self.find(mu)
        if idx is not None:
            k = idx-self.offset
            for key in wtarg:
                self.target[key][k] = copy(wtarg[key])
            return idx

        # Handle case where mu not seen
        self.mu_db.append(copy(mu))
        for key in self.target:
            if key in wtarg:
                self.target[key].append(copy(wtarg[key]))
            else:
                self.target[key].append(None)
        return self.offset+len(self.mu_db)-1

    def find(self, mu):
        """
        Return index and target of entry in database

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to find in database

        Return values
        -------------
        idx : None or int
          Index of parameter mu in database (None if doesn't exist)

        targ : None or dict of iterable of number
          All targets for corresponding entry in database
          (None if doesn't exist)

        Example
        -------
        >> idx, targ = db.find([0, 0])
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        for k, muk in enumerate(self.mu_db):
            if self.dist(muk, mu) == 0.0:
                idx = k+self.offset
                targ = dict([(s, self.target[s][k]) for s in self.target])
                return idx, targ
        return None, None

    def check(self, mu):
        """
        Check if entry in database exists

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to check if exists in database

        Return values
        -------------
        is_member : bool
          Whether mu exists in database

        Example
        -------
        >> is_member = db.check([0, 0])
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        return self.find(mu)[0] is not None

    def find_target(self, mu, wtarg):
        """
        Find target in database

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to find in database

        wtarg : str
          Target to extract

        Return values
        -------------
        targ : iterable of number
          Target corresponding to parameter mu

        Example
        -------
        >> db.find_target([0, 0], 'prim')
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        self.check_permissible_target(wtarg)
        idx, which = self.find(mu)
        return which[wtarg] if idx is not None else None

    def find_closest_target(self, mu, wtarg):
        """
        Find target in database whose parameter is closest to mu

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to find (closest) in database

        wtarg : str
          Target to extract

        Return values
        -------------
        targ : iterable of number
          Target whose parameter is closest to mu

        Example
        -------
        >> db.find_closest_target([0, 0], 'prim')
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        self.check_permissible_target(wtarg)
        dist = []
        for k, muk in enumerate(self.mu_db):
            dist.append(self.dist(muk, mu))
        kk = argmin(dist)
        return self.target[wtarg][kk]

    def countall_target(self, wtarg):
        """
        Count number of parameters with specific target

        Input arguments
        ---------------
        wtarg : str
          Target to extract
        
        Return values
        -------------
        cnt : int
          Number of parameters with specific target
                
        Example
        -------
        >> db.countall_target('prim')
        """
        self.check_permissible_target(wtarg)
        return sum([1 for x in self.target[wtarg] if x is not None])

    def getall_target(self, wtarg):
        """
        Get all entries in database with specific target

        Input arguments
        ---------------
        wtarg : str 
          Target to extract
        
        Return values
        -------------
        all_targ : list of iterable of number
          All entreis in database with specific target
                
        Example
        -------
        >> db.getall_target('prim')
        """
        self.check_permissible_target(wtarg)
        return [x for x in self.target[wtarg] if x is not None]

    def getall_target_except(self, mu, wtarg):
        """
        Get all entries in database with specific target except those
        for parameter mu

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to exclude from query

        wtarg : str 
          Target to extract
        
        Return values
        -------------
        all_targ : list of iterable of number
          All entries in database with specific target
                
        Example
        -------
        >> db.getall_target_except([0, 0], 'prim')
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        self.check_permissible_target(wtarg)
        all_targ, targ, dist = [], self.target[wtarg], self.dist
        for k in range(len(self.mu_db)):
            if targ[k] is None: continue
            if dist(self.mu_db[k], mu) == 0: continue
            all_targ.append(targ[k])
        return all_targ

    def check_target(self, mu, wtarg):
        """
        Check if specific target corresponding to parameter mu
        exists in database

        Input arguments
        ---------------
        mu : iterable of number
          Parameter to find in database

        wtarg : str 
          Target to check
        
        Return values
        ------------- 
        is_member : bool
          Whether mu/wtarg exists in database
                
        Example
        -------
        >> db.check_target([0, 0], 'prim')
        """
        if not is_type(mu, 'iter_of_number'):
            raise TypeError('mu must be iterable of number')
        self.check_permissible_target(wtarg)
        return self.find_target(mu, wtarg) is not None

    def current(self):
        """
        Get index of current (last) entry in database

        Example
        -------
        >> idx = self.current()
        """
        return self.offset+len(self.mu_db)-1

    def reset(self):
        """
        Reset database with same targets and offset as self
        with empty parameter set

        Example
        -------
        >> db_new = db.reset()
        """
        targets = [x for x in self.target]
        return Database(self.dist, targets, self.offset, self.copy)

if __name__ == '__main__':
    dist = lambda a, b: norm([a[k]-b[k] for k in range(len(a))])
    db = Database(dist=dist, targets=['prim', 'sens', 'dual'])
    db.add([0, 0], prim=[0])
    db.add([1, 0], prim=[2], sens=[0.5])
    db.add([0, 1], prim=[1], dual=[1])
    print db.mu_db, db.target
    print db.find([1, 0])
    print db.check([0, 0])
    print db.check([1, 1])
    print db.find_target([0, 1], 'dual')
    print db.find_closest_target([0.1, 0.1], 'prim')
    print db.countall_target('prim')
    print db.countall_target('dual')
    print db.getall_target('prim')
    print db.getall_target_except([0, 0], 'prim')
    print db.check_target([0, 0], 'prim')
    print db.check_target([1, 0], 'dual')
    print db.current()
