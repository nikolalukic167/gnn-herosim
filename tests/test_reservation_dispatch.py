import numpy as np
import pytest
from src.placement.radical.block_moves import scaled_problem
from src.placement.radical.dispatch_live import replay as previous
from src.placement.radical.reservation_dispatch import replay
from scripts_cosim.dispatch_reservation_witness import fixture, certificate


@pytest.mark.parametrize('seed',[145901,145902,145903])
def test_zero_reservations_preserve_old_replay(seed):
    b = scaled_problem(seed,4,4,4)
    a = b['p'].argmin(-1)
    rank = np.arange(a.size).reshape(a.shape)
    old,new = previous(b,a,rank),replay(b,a,rank,np.zeros(a.shape))
    assert old['objective']==new['objective']
    assert np.array_equal(old['ends'],new['ends'])
    assert old['events']==new['events']


def test_reservation_witness_and_fully_idle_wakeup():
    b = fixture()
    a,rank = np.array([[0,1],[1,0]]),np.arange(4).reshape(2,2)
    release = np.array([[6.,0.],[0.,0.]])
    assert previous(b,a,rank)['objective']==30
    assert replay(b,a,rank,release)['objective']==certificate(b)['best']['cost']==27
    assert replay(b,a,rank,release+5)['objective']==37


def test_invalid_reservations():
    b = fixture()
    a,rank = np.array([[0,1],[1,0]]),np.arange(4).reshape(2,2)
    for release in (np.zeros(4),np.full((2,2),-1),np.full((2,2),np.nan)):
        with pytest.raises(ValueError):
            replay(b,a,rank,release)
