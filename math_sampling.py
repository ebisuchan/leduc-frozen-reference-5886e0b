"""Public synthetic fixed-policy tests of unmodified frozen OpenSpiel samplers.

Only storage callbacks are intercepted; traversal, sampling, reach corrections,
baseline correction, regret matching and iteration remain upstream bytecode.
This is a bounded statistical diagnostic, not training or a variance proof.
"""
import base64
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

SHA = '5886e0b822bcc469cacb53a82c18ac1e8748d820'
N = 2048
SEEDS = [966281, 966282, 966283, 966284]
PROFILES = ['M1A', 'M1B']
Z_LIMIT = 4.5


def emit(label, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    print(label+'_BEGIN '+hashlib.sha256(raw).hexdigest(), flush=True)
    for offset in range(0, len(encoded), 4000):
        print(label+'_CHUNK '+encoded[offset:offset+4000], flush=True)
    print(label+'_END', flush=True)


def main(source):
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    resource.setrlimit(resource.RLIMIT_AS, (1024*1024**2, 1024*1024**2))
    source = Path(source).resolve()
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == SHA
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--'], text=True)
    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import external_sampling_mccfr as es
    from open_spiel.python.algorithms import outcome_sampling_mccfr as os
    from open_spiel.python.algorithms import mccfr
    game = pyspiel.load_game('leduc_poker', dict(players=2, action_mapping=False, suit_isomorphism=False, starting_player=0))
    states, menus = [], {}
    todo = [game.new_initial_state()]
    while todo:
        state = todo.pop()
        states.append(state)
        if not state.is_terminal() and not state.is_chance_node():
            key = state.information_state_string(state.current_player())
            actions = list(state.legal_actions())
            assert key not in menus or menus[key] == actions
            menus[key] = actions
        todo.extend(state.child(a) for a in state.legal_actions())
    assert (len(states), len(menus)) == (9457, 936)
    keys = sorted(menus)
    coordinates = [(key, action) for key in keys for action in menus[key]]
    index = {item: i for i, item in enumerate(coordinates)}
    offsets = {key: [index[(key, a)] for a in menus[key]] for key in keys}
    dimension = len(coordinates)
    projections = np.array([[1.0 if hashlib.sha256((str(j)+'|'+key+'|'+str(action)).encode()).digest()[0] & 1 else -1.0
                             for key, action in coordinates]*2 for j in range(8)])/math.sqrt(2*dimension)

    def policy(mode):
        rows = {}
        for key in keys:
            digest = hashlib.sha256((mode+'|'+key).encode()).digest()
            weights = np.array([1+digest[i] % 5 for i in range(len(menus[key]))], dtype=float)
            if mode == 'M1B':
                weights = weights**2
            rows[key] = weights/weights.sum()
        return rows

    def oracle(rows, root=None):
        regret = np.zeros(dimension)
        average_es = np.zeros(dimension)
        average_os = np.zeros(dimension)

        def walk(state, reach, chance):
            if state.is_terminal():
                return np.array(state.returns())
            if state.is_chance_node():
                return sum(prob*walk(state.child(a), reach, chance*prob) for a, prob in state.chance_outcomes())
            p = state.current_player()
            key = state.information_state_string(p)
            sigma = rows[key]
            values = []
            for a, prob in zip(menus[key], sigma):
                child_reach = list(reach)
                child_reach[p] *= prob
                values.append(walk(state.child(a), child_reach, chance))
            value = sum(prob*v for prob, v in zip(sigma, values))
            for i, coordinate in enumerate(offsets[key]):
                regret[coordinate] += chance*reach[1-p]*(values[i][p]-value[p])
                average_es[coordinate] += chance*reach[p]*sigma[i]
                # Upstream OS cancels ALL prefix sampling probability in averaging.
                average_os[coordinate] += reach[p]*sigma[i]
            return value

        walk(game.new_initial_state() if root is None else root, [1.0, 1.0], 1.0)
        return regret, average_es, average_os

    class Capture:
        def __init__(self, rows):
            super().__init__(game)
            self.delta = np.zeros(2*dimension)
            # Regret matching of fixed positive weights gives the target profile.
            self._infostates = {k: [np.array(rows[k]), np.zeros(len(rows[k]))] for k in keys}

        def _add_regret(self, key, action_idx, amount):
            self.delta[offsets[key][action_idx]] += amount

        def _add_avstrat(self, key, action_idx, amount):
            self.delta[dimension+offsets[key][action_idx]] += amount

    class FixedES(Capture, es.ExternalSamplingSolver):
        pass

    class FixedOS(Capture, os.OutcomeSamplingSolver):
        pass

    result = dict(commit=SHA, numpy_version=np.__version__, module_sha256=hashlib.sha256(Path(pyspiel.__file__).read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  upstream_source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                          (Path(es.__file__), Path(os.__file__), Path(mccfr.__file__))},
                  sample_count_per_case=N, seeds=SEEDS, profiles=PROFILES, projection_z_limit=Z_LIMIT,
                  primary_test='8 predeclared signed linear projections per case; abs(mean-target) <= 4.5*SE + 1e-10; normal approximation, not a finite-sample confidence guarantee',
                  coordinate_fields=['regret', 'average'], coordinates=coordinates, dimension_per_field=dimension,
                  training=False, ast_rewriting=False, fixed_positive_policy_weights=True,
                  unchanged_upstream_methods=['ExternalSamplingSolver.iteration/_update_regrets', 'OutcomeSamplingSolver.iteration/_episode/_baseline_corrected_child_value', 'MCCFRSolverBase._regret_matching'],
                  instrumentation='Only _add_regret and _add_avstrat storage redirected to per-trial delta; no feedback to fixed policy',
                  cases=[], finite_tapes=[], failures=[])
    try:
        for mode in PROFILES:
            rows = policy(mode)
            rg, avg_es, avg_os = oracle(rows)
            for kind, cls, avg in (('ES', FixedES, avg_es), ('OS', FixedOS, avg_os)):
                solver = cls(rows)
                assert solver._regret_matching.__func__ is mccfr.MCCFRSolverBase._regret_matching
                if kind == 'ES':
                    assert solver._update_regrets.__func__ is es.ExternalSamplingSolver._update_regrets
                else:
                    assert solver._episode.__func__ is os.OutcomeSamplingSolver._episode
                expected = np.concatenate((rg, avg))
                total = np.zeros(2*dimension)
                squares = np.zeros(2*dimension)
                nonzero = np.zeros(2*dimension, dtype=int)
                projected = []
                case = dict(profile=mode, algorithm=kind, policies={k: rows[k].tolist() for k in keys},
                            exact_regret=rg.tolist(), exact_average=avg.tolist(), samples=[], batches=[])
                result['cases'].append(case)
                for batch, seed in enumerate(SEEDS):
                    np.random.seed(seed + (100 if mode == 'M1B' else 0) + (1000 if kind == 'OS' else 0))
                    batch_start = time.process_time()
                    for trial in range(N//len(SEEDS)):
                        solver.delta.fill(0)
                        solver.iteration()  # REAL upstream random sampling; no AST transform.
                        value = solver.delta
                        assert np.all(np.isfinite(value))
                        total += value
                        squares += value*value
                        nz = np.flatnonzero(value)
                        nonzero[nz] += 1
                        projected.append(projections@value)
                        # Retain every sample, including sparse zero coordinates by convention.
                        case['samples'].append([nz.tolist(), value[nz].tolist()])
                    batch_record = dict(batch=batch, seed=seed + (100 if mode == 'M1B' else 0) + (1000 if kind == 'OS' else 0),
                                        samples=N//len(SEEDS), cpu_seconds=time.process_time()-batch_start)
                    case['batches'].append(batch_record)
                    print('M1_PROGRESS '+json.dumps(dict(profile=mode, algorithm=kind, **batch_record)), flush=True)
                for key in keys:
                    assert np.array_equal(solver._infostates[key][0], rows[key])
                mean = total/N
                variance = np.maximum(0, (squares-total*total/N)/(N-1))
                se = np.sqrt(variance/N)
                samples = np.asarray(projected)
                target = projections@expected
                proj_mean = samples.mean(axis=0)
                proj_se = samples.std(axis=0, ddof=1)/math.sqrt(N)
                error = np.abs(proj_mean-target)
                passed = bool(np.all(error <= Z_LIMIT*proj_se+1e-10))
                case.update(n=N, empirical_mean=mean.tolist(), standard_error=se.tolist(), nonzero_sample_counts=nonzero.tolist(),
                            exact_average_chance_weighted_local_convention=avg_es.tolist(),
                            projection_expected=target.tolist(), projection_mean=proj_mean.tolist(), projection_standard_error=proj_se.tolist(),
                            projection_z=[float(e/s) if s else None for e, s in zip(error, proj_se)],
                            unobserved_nonzero_target_coordinates=int(np.sum((nonzero == 0) & (np.abs(expected)>1e-12))),
                            status='PASS BOUNDED EMPIRICAL CONSISTENCY' if passed else 'FAIL EMPIRICAL CONSISTENCY',
                            average_convention='ES chance-weighted own reach' if kind == 'ES' else 'OS chance-cancelled own reach; multiply by per-infoset chance mass to compare local cumulative averages')
                if not passed:
                    result['failures'].append(dict(profile=mode, algorithm=kind, reason='predeclared projection threshold exceeded'))

        # Small finite conditional random tape: actual ES traversal with choice replay.
        rows = policy('M1A')
        root = game.new_initial_state()
        prefix = [0, 2, 1, 1, 1]
        for action in prefix:
            assert action in root.legal_actions()
            root.apply_action(action)
        conditional = oracle(rows, root)
        finite_mean = np.zeros(2*dimension)
        original_choice = np.random.choice

        class NeedChoice(Exception):
            def __init__(self, options, probabilities):
                self.options, self.probabilities = options, probabilities

        try:
            for player in (0, 1):
                pending = [((), 1.0)]
                leaf_count = expanded = 0
                mass = 0.0
                player_mean = np.zeros(2*dimension)
                traces = []
                while pending:
                    tape, probability = pending.pop()
                    solver = FixedES(rows)
                    cursor = 0

                    def replay(options, p=None, **kwargs):
                        nonlocal cursor
                        assert not kwargs
                        options = list(range(options)) if isinstance(options, int) else list(options)
                        probs = [1/len(options)]*len(options) if p is None else list(p)
                        if cursor == len(tape):
                            raise NeedChoice(options, probs)
                        value = tape[cursor]
                        cursor += 1
                        assert value in options
                        return value

                    np.random.choice = replay
                    try:
                        solver._update_regrets(root.clone(), player)
                    except NeedChoice as need:
                        expanded += 1
                        assert expanded <= 2048
                        pending.extend((tape+(int(a),), probability*float(p)) for a, p in zip(need.options, need.probabilities) if p > 0)
                    else:
                        assert cursor == len(tape)
                        leaf_count += 1
                        mass += probability
                        player_mean += probability*solver.delta
                        nz = np.flatnonzero(solver.delta)
                        traces.append(dict(tape=tape, probability=probability, indices=nz.tolist(), increments=solver.delta[nz].tolist()))
                finite_mean += player_mean
                result['finite_tapes'].append(dict(player=player, conditional_prefix=prefix, leaves=leaf_count,
                                                   probability_mass=mass, traces=traces))
                assert abs(mass-1) < 1e-12
        finally:
            np.random.choice = original_choice
        finite_error = float(np.max(np.abs(finite_mean-np.concatenate(conditional[:2]))))
        result['finite_tape_max_error'] = finite_error
        if finite_error > 1e-12:
            result['failures'].append(dict(reason='finite conditional ES random tape expectation mismatch', error=finite_error))
        result['status'] = 'PASS BOUNDED M1 TESTS' if not result['failures'] else 'FAIL M1 TESTS'
    except Exception:
        result['status'] = 'FAIL M1 EXECUTION'
        result['failures'].append(dict(exception=traceback.format_exc()))
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['resource'] = dict(cpu_seconds=usage.ru_utime+usage.ru_stime, wall_seconds=time.monotonic()-started,
                                  maxrss_kib=usage.ru_maxrss, cpu_limit_seconds=120, wall_limit_seconds=150,
                                  address_space_limit_mib=1024)
        result['limitations'] = ['Statistical evidence is bounded; no variance/convergence theorem',
                                 'Finite tape exhaustive test is conditional late-round ES only',
                                 'Full coordinate means/SE/coverage retained; primary gate uses 8 fixed projections',
                                 'No candidate private implementation uploaded or executed in CI',
                                 'M2 transform audit and final admission remain with Copilot']
        emit('LEDUC_M1', result)
        print('M1_SUMMARY '+json.dumps(dict(status=result['status'], resource=result['resource'], failures=len(result['failures']))), flush=True)
    if not result['status'].startswith('PASS'):
        raise SystemExit(1)


if __name__ == '__main__':
    main(sys.argv[1])
