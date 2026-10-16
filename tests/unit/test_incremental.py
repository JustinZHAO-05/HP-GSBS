import numpy as np
import pytest
from hpgsbs.core.incremental_tdistance import IncrementalTDistance
from hpgsbs.core.tdistance import TDistance


@pytest.mark.parametrize("dmin",[1,2,5,20])
def test_split_double_delete_insert_and_distant_move(dmin):
    x=np.random.default_rng(76).normal(size=(40,8))
    td=IncrementalTDistance(x,dmin); oracle=TDistance(x,dmin)
    # Single, double, delete, insert, distant move, competing/rollback partitions.
    path=[[12],[12,20,32],[20,32],[5,20,32],[5,7,20],[5,7,20,27],
          [5,7,20],[2,5,7,20],[2,5,7,20,39]]
    for cuts in path*3:
        b=np.zeros(40,bool); b[cuts]=True
        for actual,expected in zip(td.stats(b),oracle.stats(b)):
            np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-12)
        assert len(td.moments)==len(cuts)+1
        assert len(td.regions)==2*len(cuts)+1


def test_single_double_update_formulas():
    x=np.random.default_rng(13).normal(size=(25,5)); td=IncrementalTDistance(x)
    phi=lambda a,b,c,d:td.cross(td.moment(a,b),td.moment(c,d))
    old=np.zeros(25,bool); old[[4,21]]=True
    w,a=td.stats(old)
    single=old.copy(); single[10]=True
    sw,sa=td.stats(single)
    np.testing.assert_allclose(sw-w,-phi(4,10,10,21),atol=1e-12)
    np.testing.assert_allclose(sa-a,phi(4,10,10,21)-phi(0,4,10,21)-phi(4,10,21,25),atol=1e-12)
    double=old.copy(); double[[10,15]]=True
    dw,da=td.stats(double)
    np.testing.assert_allclose(dw-w,-phi(4,10,10,15)-phi(4,10,15,21)-phi(10,15,15,21),atol=1e-12)
    np.testing.assert_allclose(da-a,phi(4,10,10,15)+phi(10,15,15,21)-phi(0,4,10,15)-phi(0,4,15,21)-phi(4,10,21,25)-phi(10,15,21,25),atol=1e-12)
