"""Acceptance cannot be rescued by pooling repeated traces or hiding a bad tail."""
import copy
import json
import hashlib
import pytest

from scripts_cosim.peer_lookahead_validate import (
    ROOT, evaluate, fresh_workload, healthy, infrastructure_fingerprint, verify_sources,
)


def config():
    return json.loads((ROOT/'experiments/peer_lookahead_validation_v1.json').read_text())


def test_validation_uses_distinct_infrastructures_and_new_live_sources():
    c=config()
    assert len(c['sources'])==16
    assert c['source_fingerprint_contract'] == 'physical_infrastructure_v2'
    assert len({s['physical_infrastructure_sha256'] for s in c['source_manifest']})==16
    assert not set(c['sources'])&{'ds_00000','ds_00064','ds_00128'}
    assert sum(not s['seen_in_paper_screen'] for s in c['source_manifest'])==10


def test_fresh_workload_is_reproducible_and_time_scaling_preserves_manifest():
    c=config();source={'config':{'workload':{'events':[{'node_name':'client0'}]}}}
    a=fresh_workload(source,c,1.,44101)
    b=fresh_workload(source,c,2.,44101)
    assert a==fresh_workload(source,c,1.,44101)
    assert a['peer_exchange']==b['peer_exchange']
    assert all(y['timestamp']==2*x['timestamp'] for x,y in zip(a['events'],b['events']))
    assert all(i//8==j//8 for i,j,_ in a['peer_exchange'])
    assert a!=fresh_workload(source,c,1.,44102)


def test_queue_growth_can_fail_even_below_queue_share_limit():
    bars=config()['acceptance']
    s={'queue':70.,'total_rtt':100.,'quarter_mean_rtt':[1.,2.,3.,5.]}
    assert not healthy(s,bars)['pass']
    s['quarter_mean_rtt']=[1.,2.,2.,2.]
    assert healthy(s,bars)['pass']


def test_bad_tail_blocks_promotion_despite_positive_median():
    c=config()
    results=[{'source':str(i),'error':None,'traces':[
        {'seed':j,'arms':{'immediate':{'total_rtt':100.},'peer_mass':{'total_rtt':100.},
                         'group_first_two_hop':{'total_rtt':80.}}} for j in (1,2)]} for i in range(16)]
    assert evaluate(results,c)['verdict']=='ADVANCE'
    bad=copy.deepcopy(results)
    bad[0]['traces'][0]['arms']['group_first_two_hop']['total_rtt']=111.
    verdict=evaluate(bad,c)
    assert verdict['verdict']=='DO-NOT-ADVANCE'
    assert not verdict['comparisons']['peer_mass']['checks']['regression_bound']
    assert evaluate(results[:8],c)['verdict']=='DO-NOT-ADVANCE'


def manifest_fixture(tmp_path):
    cfg = {'source_corpus': str(tmp_path), 'sources': ['a', 'b'], 'source_manifest': [],
           'source_fingerprint_contract': 'physical_infrastructure_v2'}
    for i, name in enumerate(cfg['sources']):
        obj = {'config': {'infrastructure': {'network': {'bandwidth': 100+i}, 'nodes': []}}}
        raw = json.dumps(obj).encode()
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'optimal_result.json').write_bytes(raw)
        cfg['source_manifest'].append({'source': name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                                      'physical_infrastructure_sha256': infrastructure_fingerprint(obj)})
    return cfg


def test_preflight_verifies_actual_bytes(tmp_path):
    cfg = manifest_fixture(tmp_path)
    assert len(verify_sources(cfg)) == 2
    (tmp_path / 'a' / 'optimal_result.json').write_text('{}')
    with pytest.raises(ValueError, match='source fingerprint mismatch'):
        verify_sources(cfg)


def test_preflight_rejects_duplicate_actual_infrastructure_even_with_honest_hashes(tmp_path):
    cfg = manifest_fixture(tmp_path)
    raw = (tmp_path / 'a' / 'optimal_result.json').read_bytes()
    (tmp_path / 'b' / 'optimal_result.json').write_bytes(raw)
    cfg['source_manifest'][1].update(source_sha256=hashlib.sha256(raw).hexdigest(),
                                    physical_infrastructure_sha256=infrastructure_fingerprint(json.loads(raw)))
    with pytest.raises(ValueError, match='duplicate actual'):
        verify_sources(cfg)


def test_preflight_rejects_false_identity_and_repeated_source_names(tmp_path):
    cfg = manifest_fixture(tmp_path)
    cfg['source_manifest'][0]['physical_infrastructure_sha256'] = 'fake'
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        verify_sources(cfg)
    cfg['sources'] = ['a', 'a']
    with pytest.raises(ValueError, match='duplicate sources'):
        verify_sources(cfg)


def test_invalid_manifest_stops_before_worker_creation(tmp_path, monkeypatch):
    from scripts_cosim import peer_lookahead_validate as validator
    cfg = manifest_fixture(tmp_path)
    cfg['source_manifest'][0]['source_sha256'] = 'mismatch'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr('sys.argv', ['validate', '--config', str(path), '--out-dir', str(tmp_path / 'out')])
    monkeypatch.delenv('HEROSIM_PG_DRAIN_CONTRACT', raising=False)
    def forbidden_pool(*args, **kwargs):
        pytest.fail('workers started before source verification')
    monkeypatch.setattr(validator, 'ProcessPoolExecutor', forbidden_pool)
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        validator.main()
    assert not (tmp_path / 'out').exists()


def test_worker_rechecks_source_before_simulation(tmp_path, monkeypatch):
    from scripts_cosim import peer_lookahead_validate as validator
    cfg = manifest_fixture(tmp_path)
    verify_sources(cfg)
    path = tmp_path / 'a' / 'optimal_result.json'
    obj = json.loads(path.read_text())
    obj['config']['infrastructure']['network']['bandwidth'] += 1
    path.write_text(json.dumps(obj))
    monkeypatch.setattr(validator, 'environment', lambda: None)
    def forbidden_run(*args, **kwargs):
        pytest.fail('simulation started with mutated source')
    monkeypatch.setattr(validator, 'run', forbidden_run)
    with pytest.raises(ValueError, match='source changed after preflight'):
        validator.one_source((cfg, 'a', tmp_path / 'out'))


@pytest.mark.parametrize('module', ['peer_lookahead_live_probe', 'peer_lookahead_trace', 'peer_lookahead_validate'])
def test_closed_probe_cli_does_not_silently_reset_new_contract(module, monkeypatch):
    import importlib
    monkeypatch.setenv('HEROSIM_PG_DRAIN_CONTRACT', 'availability_v2')
    with pytest.raises(ValueError, match='requires backlog_only_v1'):
        importlib.import_module(f'scripts_cosim.{module}').main()
