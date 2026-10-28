"""Export only synthetic public OpenSpiel Leduc states; no project imports."""
import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

SHA = '5886e0b822bcc469cacb53a82c18ac1e8748d820'
PARAMETERS = dict(players=2, action_mapping=False, suit_isomorphism=False, starting_player=0)


def main():
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    source = Path(sys.argv[1]).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != SHA:
        raise RuntimeError('Frozen full commit mismatch')
    changes = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--'], text=True)
    if changes:
        raise RuntimeError('Tracked upstream source modified')
    import pyspiel
    module = Path(pyspiel.__file__).resolve()
    if not module.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('pyspiel did not load from isolated build venv')
    game = pyspiel.load_game('leduc_poker', PARAMETERS)
    rows, keys, menus = [], set(), {}
    stack = [((), game.new_initial_state())]
    while stack:
        path, state = stack.pop()
        terminal, chance = state.is_terminal(), state.is_chance_node()
        actor = state.current_player()
        actions = list(state.legal_actions())
        row = dict(path=path, terminal=terminal, chance=chance, actor=actor,
                   actions=actions, transitions=[dict(action=a, child=path+(a,)) for a in actions],
                   private=list(state.get_private_cards()), public=state.public_card(),
                   round=state.round(), round1=list(state.round1()), round2=list(state.round2()),
                   money=list(state.money()), pot=state.pot(), returns=list(state.returns()),
                   chance_outcomes=list(state.chance_outcomes()) if chance else [],
                   infosets={}, action_labels={str(a): state.action_to_string(actor, a) for a in actions})
        if not terminal and not chance:
            row['infosets'][str(actor)] = state.information_state_string(actor)
            key = (actor, row['infosets'][str(actor)])
            if key in menus and menus[key] != actions:
                raise RuntimeError('Reference infoset has inconsistent action menu')
            menus[key] = actions
            keys.add(key)
        if terminal and (not all(math.isfinite(v) for v in row['returns']) or abs(sum(row['returns'])) > 1e-12):
            raise RuntimeError('Invalid terminal settlement')
        rows.append(row)
        if len(rows) > 9457:
            raise RuntimeError('Tree cap exceeded')
        stack.extend((path+(a,), state.child(a)) for a in reversed(actions))
    counts = dict(nodes=len(rows), terminals=sum(r['terminal'] for r in rows), infosets=len(keys))
    if counts != dict(nodes=9457, terminals=5520, infosets=936):
        raise RuntimeError('Unexpected public reference counts: '+repr(counts))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if usage.ru_maxrss > 256*1024 or usage.ru_utime+usage.ru_stime > 30:
        raise RuntimeError('Export resource cap exceeded')
    metadata = dict(schema=1, status='PUBLIC FROZEN REFERENCE EXPORTED', commit=commit,
                    parameters=PARAMETERS, counts=counts, python=sys.version, platform=platform.platform(),
                    module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                    exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cpu_seconds=usage.ru_utime+usage.ru_stime, maxrss_kib=usage.ru_maxrss,
                    wall_seconds=time.monotonic()-started, training=False,
                    run_id=os.environ.get('GITHUB_RUN_ID'), workflow_commit=os.environ.get('GITHUB_SHA'))
    payload = json.dumps(dict(metadata=metadata, states=rows), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    compressed = gzip.compress(payload, mtime=0)
    # Log transport avoids artifact/cache storage. Retrieve via authenticated run logs.
    print('LEDUC_EXPORT_BEGIN '+hashlib.sha256(payload).hexdigest(), flush=True)
    encoded = base64.b64encode(compressed).decode()
    for offset in range(0, len(encoded), 4000):
        print('LEDUC_EXPORT_CHUNK '+encoded[offset:offset+4000], flush=True)
    print('LEDUC_EXPORT_END', flush=True)
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
