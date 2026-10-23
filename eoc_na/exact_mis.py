"""Exact MIS solver from the historical MIS_4_mDS.py implementation."""
import time

def maximum_independent_set_exact_fast(G, time_limit=None):
    """Exact Maximum Independent Set via Maximum Clique on complement graph."""

    start_time = time.perf_counter()

    # Helpful ordering: low-degree vertices in G are promising for MIS
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v))
    n = len(nodes)

    idx = {v: i for i, v in enumerate(nodes)}
    rev = {i: v for v, i in idx.items()}

    full = (1 << n) - 1

    # Original graph adjacency as bitmasks
    orig_adj = [0] * n

    for u, v in G.edges():
        i = idx[u]
        j = idx[v]
        orig_adj[i] |= 1 << j
        orig_adj[j] |= 1 << i

    # Complement graph adjacency as bitmasks
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = full & ~(orig_adj[i] | (1 << i))

    best_bits = 0
    best_size = 0
    calls = 0

    def _check_time():
        if time_limit is not None:
            if time.perf_counter() - start_time > time_limit:
                raise TimeoutError

    def _greedy_color_sort(P):
        """Greedy coloring of the candidate subgraph in the complement graph."""
        order = []
        colors = []

        U = P
        color = 0

        while U:
            color += 1
            Q = U

            while Q:
                v_bit = Q & -Q
                v = v_bit.bit_length() - 1

                order.append(v)
                colors.append(color)

                U &= ~v_bit
                Q &= ~v_bit

                # Same color class must be independent in complement,
                # so remove complement-neighbors of v from Q.
                Q &= ~comp_adj[v]

        return order, colors

    def _expand(C_bits, C_size, P):
        nonlocal best_bits, best_size, calls

        calls += 1
        if calls % 4096 == 0:
            _check_time()

        if P == 0:
            if C_size > best_size:
                best_size = C_size
                best_bits = C_bits
            return

        # Cheap bound
        if C_size + P.bit_count() <= best_size:
            return

        order, colors = _greedy_color_sort(P)

        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            v_bit = 1 << v

            # Strong coloring bound
            if C_size + colors[i] <= best_size:
                return

            if not (P & v_bit):
                continue

            _expand(
                C_bits | v_bit,
                C_size + 1,
                P & comp_adj[v],
            )

            P &= ~v_bit

            if C_size + P.bit_count() <= best_size:
                return

    try:
        _expand(0, 0, full)
        exact = True
    except TimeoutError:
        exact = False

    best_set = {
        rev[i]
        for i in range(n)
        if best_bits & (1 << i)
    }

    return best_set, exact
