"""The recovery checkpoint owns this RNG; choose before any rollout."""
def shallow_jobs_for_update(indices, rng, depths):
    assert tuple(depths)==(1,2,3) and len(indices)>0
    depth=rng.choice(depths)
    return [(i,depth) for i in indices]
