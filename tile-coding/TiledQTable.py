
import numpy as np

from tile_encode import create_tilings, tile_encode


class QTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size):
        """Initialize Q-table.

        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        self.state_size = state_size
        self.action_size = action_size

        # TODO: Create Q-table, initialize all Q-values to zero
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
        print("QTable(): size =", self.q_table.shape)


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme."""

    def __init__(self, low, high, tiling_specs, action_size, tiling_grids=None, tilings=None):
        """Create tilings and initialize internal Q-table(s).

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        if tiling_specs is None:
            # auto random tiling_specs
            intervals = (high-low) / tilings
            tiling_specs = []
            for i in range(tiling_grids):
                tiling_specs.append((tuple([tilings for i in range(len(low))]),
                                     tuple([(np.random.rand()-0.5)*intr for intr in intervals ]) ))

        self.tiling_specs = tiling_specs
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [QTable(state_size, self.action_size) for state_size in self.state_sizes]
        print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

    def get_tiling_specs(self):
        return self.tiling_specs

    def get(self, state, action):
        """Get Q-value for given <state, action> pair.

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.

        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        # TODO: Encode state to get tile indices
        encoded_state = tile_encode(state, self.tilings)
        # TODO: Retrieve q-value for each tiling, and return their average
        qv = 0
        for i in range(len(encoded_state)):
            qv = qv + self.q_tables[i].q_table[tuple(encoded_state[i] + (action,))]
        qv = qv / len(encoded_state)
        return qv

    def update(self, state, action, value, alpha=0.1):
        """Soft-update Q-value for given <state, action> pair to value.

        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """
        # TODO: Encode state to get tile indices
        encoded_state = tile_encode(state, self.tilings)

        # TODO: Update q-value for each tiling by update factor alpha
        for i in range(len(encoded_state)):
            self.q_tables[i].q_table[tuple(encoded_state[i] + (action,))] = self.q_tables[i].q_table[
                                                                                tuple(encoded_state[i] + (action,))] * (
                                                                                    1 - alpha) + value * alpha
